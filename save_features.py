#some code adapted from https://github.com/YuemingJin/MTRCNet-CL
# and https://github.com/YuemingJin/TMRNet

import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import pickle
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import logging
import sys

from models.spatial_extractor import get_spatial_extractor
from utils.utils import get_start_idx, get_data, SeqSampler



def save_features(config, use_gpu, exp_path, fold):
    num_gpu = torch.cuda.device_count()
    use_gpu = torch.cuda.is_available() and use_gpu
    device = torch.device("cuda:0" if use_gpu else "cpu")

    num_workers = config.general["num_workers"]
    dataset = config.general["dataset"]
    img_path = config.dataset_dirs[dataset]["img_path"]
    label_path = config.dataset_dirs[dataset]["label_path"]
    get_feats = config.save_features["get_feats"]
    sequence_length = config.save_features["sequence_length"]
    val_batch_size = config.save_features["val_batch_size"]
    backbone = config.spatial_extractor["backbone"]

    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S :")
    fh = logging.FileHandler(exp_path + "/logs/save_features.txt")
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # Log configs for saving features
    logging.info(f"number of gpu            :{num_gpu}")
    logging.info(f"number of workers        :{num_workers}")
    logging.info(f"dataset                  :{dataset}")
    logging.info(f"image directory path     :{img_path}")
    logging.info(f"label directory path     :{label_path}")
    logging.info(f"get features             :{get_feats}")
    logging.info(f"validation batch size    :{val_batch_size}")
    logging.info(f"sequence length          :{sequence_length}")
    logging.info(f"backbone                 :{backbone}")

    # TensorBoard
    writer = SummaryWriter(exp_path + "/runs/non-local/pretrained_lr5e-7_L40_2fc_copy/")

    train_dataset, train_num_each, val_dataset, val_num_each = get_data(img_path, label_path, fold, sequence_length, get_feats)
    train_dataset, train_num_each, val_dataset, val_num_each = (train_dataset), (train_num_each), (val_dataset), (val_num_each)

    (train_num_each), (val_dataset), (val_num_each) = train_num_each, val_dataset, val_num_each

    (train_dataset, train_dataset_feats) = train_dataset

    train_start_idx = get_start_idx(sequence_length, train_num_each)
    val_start_idx = get_start_idx(sequence_length, val_num_each)

    train_start_idx_feats = get_start_idx(sequence_length, train_num_each)
    val_start_idx_feats = get_start_idx(sequence_length, val_num_each)

    num_train = len(train_start_idx)
    num_val = len(val_start_idx)

    num_train_feats = len(train_start_idx_feats)
    num_val_feats = len(val_start_idx_feats)

    train_idx = []
    for i in range(num_train):
        for j in range(sequence_length):
            train_idx.append(train_start_idx[i] + j)

    val_idx = []
    for i in range(num_val):
        for j in range(sequence_length):
            val_idx.append(val_start_idx[i] + j)

    train_idx_feats = []
    for i in range(num_train_feats):
        for j in range(sequence_length):
            train_idx_feats.append(train_start_idx_feats[i] + j)

    val_idx_feats = []
    for i in range(num_val_feats):
        for j in range(sequence_length):
            val_idx_feats.append(val_start_idx_feats[i] + j)

    num_train_all = len(train_idx)
    num_val_all = len(val_idx)

    logging.info(f'num of all train use: {num_train_all}')
    logging.info(f'num of all valid use: {num_val_all}')

    feats_train = np.zeros(shape=(0, 512))
    feats_val = np.zeros(shape=(0, 512))
    feats_labels_val = np.zeros(shape=(0))
    feats_labels_train = np.zeros(shape=(0))
    logging.info("loading features!>.........")

    train_feature_loader = DataLoader(
        train_dataset_feats,
        batch_size=val_batch_size,
        sampler=SeqSampler(train_dataset_feats, train_idx_feats),
        num_workers=num_workers,
        pin_memory=False
    )
    val_feature_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=SeqSampler(val_dataset, val_idx_feats),
        num_workers=num_workers,
        pin_memory=False
    )

    model = get_spatial_extractor(backbone)
    model.load_state_dict(torch.load(exp_path + "/checkpoints/spatial_extractor_best.pth"))

    def get_parameter_number(net):
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return trainable_num

    total_papa_num = 0
    total_papa_num += get_parameter_number(model)

    if use_gpu:
        model = DataParallel(model)
        model.to(device)

    for params in model.parameters():
        params.requires_grad = False

    model.eval()

    with torch.no_grad():
        for data in train_feature_loader:
            if use_gpu:
                inputs, labels_phase = data[0].to(device), data[1].to(device)
            else:
                inputs, labels_phase = data[0], data[1]

            inputs = inputs.view(-1, sequence_length, 3, 224, 224)
            outputs_feature = model.forward(inputs).data.cpu().numpy()

            feats_train = np.concatenate((feats_train, outputs_feature), axis=0)
            
            feats_labels_train = np.concatenate((feats_labels_train, labels_phase.cpu()), axis=0)

            logging.info(f'train feature length: {len(feats_train)}')

        for data in val_feature_loader:
            if use_gpu:
                inputs, labels_phase = data[0].to(device), data[1].to(device)
            else:
                inputs, labels_phase = data[0], data[1]

            inputs = inputs.view(-1, sequence_length, 3, 224, 224)
            outputs_feature = model.forward(inputs).data.cpu().numpy()

            feats_val = np.concatenate((feats_val, outputs_feature), axis=0)

            feats_labels_val = np.concatenate((feats_labels_val, labels_phase.cpu()), axis=0)

            logging.info(f'val feature length: {len(feats_val)}')

    logging.info('finish!')
    feats_train = np.array(feats_train)
    feats_val = np.array(feats_val)
    feats_labels_val = np.array(feats_labels_val)
    feats_labels_train = np.array(feats_labels_train)
    with open(exp_path + "/features/features_train.pkl", 'wb') as f:
        pickle.dump(feats_train, f)

    with open(exp_path + "/features/features_val.pkl", 'wb') as f:
        pickle.dump(feats_val, f)
    
    with open(exp_path + "/features/val_labels.pkl", 'wb') as f:
        pickle.dump(feats_labels_val, f)

    with open(exp_path + "/features/train_labels", 'wb') as f:
        pickle.dump(feats_labels_train, f)
        
    logging.info('Done')
    print()



# if __name__ == "__main__":
#     save_features()
