# some code adapted from https://github.com/YuemingJin/MTRCNet-CL
# and https://github.com/YuemingJin/TMRNet

import pandas as pd
from sklearn.utils import class_weight
import logging
import sys
import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import time
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics

from models.spatial_extractor import get_spatial_extractor
from models.ltcontext import LTC
from utils.utils import get_start_idx, get_data, get_long_feature, get_data_ltc, SeqSampler
from loss.centerloss import CenterLoss




def train_spatial_extractor(config, use_gpu, exp_path, fold):
    num_gpu = torch.cuda.device_count()
    use_gpu = torch.cuda.is_available() and use_gpu
    device = torch.device("cuda:0" if use_gpu else "cpu")

    num_workers = config.general["num_workers"]
    dataset = config.general["dataset"]
    img_path = config.dataset_dirs[dataset]["img_path"]
    label_path = config.dataset_dirs[dataset]["label_path"]
    output_feats = config.spatial_extractor["output_feats"]
    epochs = config.general["epochs"]
    train_batch_size = config.spatial_extractor["train_batch_size"]
    val_batch_size = config.spatial_extractor["val_batch_size"]
    backbone = config.spatial_extractor["backbone"]
    sequence_length = config.spatial_extractor["sequence_length"]
    sgd_lr = config.spatial_extractor["sgd_lr"]
    sgd_momentum = config.spatial_extractor["sgd_momentum"]
    sgd_weight_decay = config.spatial_extractor["sgd_weight_decay"]
    sgd_dampening = config.spatial_extractor["sgd_dampening"]
    use_nesterov = config.spatial_extractor["use_nesterov"]
    cl_lambda = config.spatial_extractor["cl_lambda"]
    cl_lr = config.spatial_extractor["cl_lr"]

    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout,level=logging.INFO,format=log_format,datefmt="%m/%d %I:%M:%S :")
    fh = logging.FileHandler(exp_path + "/logs/train_spatial_extractor.txt")
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # Log configs for spatial extractor training
    logging.info(f"number of gpu                :{num_gpu}")
    logging.info(f"number of workers            :{num_workers}")
    logging.info(f"dataset                      :{dataset}")
    logging.info(f"image directory path         :{img_path}")
    logging.info(f"label directory path         :{label_path}")
    logging.info(f"get features                 :{output_feats}")
    logging.info(f"number of epochs             :{epochs}")
    logging.info(f"train batch size             :{train_batch_size}")
    logging.info(f"validation batch size        :{val_batch_size}")
    logging.info(f"backbone                     :{backbone}")
    logging.info(f"sequence length              :{sequence_length}")
    logging.info(f"sgd learning rate            :{sgd_lr}")
    logging.info(f"sgd momentum                 :{sgd_momentum}")
    logging.info(f"sgd weight decay             :{sgd_weight_decay}")
    logging.info(f"sgd dampening                :{sgd_dampening}")
    logging.info(f"use nesterov                 :{use_nesterov}")
    logging.info(f"center loss lambda           :{cl_lambda}")
    logging.info(f"center loss learning rate    :{cl_lr}")

    # TensorBoard
    writer = SummaryWriter(exp_path + "/runs/spatial_extractor_training/")

    train_dataset, train_num_each, val_dataset, val_num_each = get_data(img_path, label_path, fold, sequence_length)

    (train_dataset), (train_num_each), (val_dataset), (val_num_each) = (
        train_dataset,
        train_num_each,
        val_dataset,
        val_num_each,
    )

    train_start_idx = get_start_idx(sequence_length, train_num_each)
    val_start_idx = get_start_idx(sequence_length, val_num_each)

    num_train = len(train_start_idx)
    num_val = len(val_start_idx)

    train_idx = []
    for i in range(num_train):
        for j in range(sequence_length):
            train_idx.append(train_start_idx[i] + j)

    val_idx = []
    for i in range(num_val):
        for j in range(sequence_length):
            val_idx.append(val_start_idx[i] + j)

    num_train_all = len(train_idx)
    num_val_all = len(val_idx)

    logging.info(f"number of all train used: {num_train_all}")
    logging.info(f"number of all valid used: {num_val_all}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=SeqSampler(val_dataset, val_idx),
        num_workers=num_workers,
        pin_memory=False,
    )

    model = get_spatial_extractor(backbone, output_feats)

    if use_gpu:
        model = DataParallel(model)
        model.to(device)

    df_train = pd.read_csv(label_path + f"{fold}train.csv")
    df_val = pd.read_csv(label_path + f"{fold}val.csv")

    labels_train = df_train["class"].tolist()
    labels_val = df_val["class"].tolist()
    labels_all = labels_train + labels_val

    weights_train = class_weight.compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2, 3, 4]), y=labels_all)
    logging.info(f"train weights: {weights_train}")

    criterion_phase = nn.CrossEntropyLoss(size_average=False, weight=torch.from_numpy(weights_train).float().to(device))
    criterion_center = CenterLoss(num_classes=5, feat_dim=512)
    optimizer = optim.SGD(
        model.parameters(),
        lr=sgd_lr,
        momentum=sgd_momentum,
        dampening=sgd_dampening,
        weight_decay=sgd_weight_decay,
        nesterov=use_nesterov,
    )
    optimizer_center = optim.SGD(criterion_center.parameters(), lr=cl_lr)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    best_model_wts = copy.deepcopy(model.module.state_dict())
    best_val_accuracy_phase = 0.0
    correspond_train_acc_phase = 0.0
    best_epoch = 0

    # Record best stats
    best_tr_loss = 0.0
    best_tr_acc = 0.0
    best_v_loss = 0.0
    best_v_acc = 0.0
    best_v_pr = 0.0
    best_v_re = 0.0
    best_v_ji = 0.0
    best_v_pr_each = 0.0
    best_v_re_each = 0.0

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        np.random.shuffle(train_start_idx)
        train_idx = []
        for i in range(num_train):
            for j in range(sequence_length):
                train_idx.append(train_start_idx[i] + j)

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=SeqSampler(train_dataset, train_idx),
            num_workers=num_workers,
            pin_memory=False,
        )

        # Sets the module in training mode.
        model.train()
        train_loss_phase = 0.0
        train_corrects_phase = 0
        batch_progress = 0.0
        running_loss_phase = 0.0
        minibatch_correct_phase = 0.0
        train_start_time = time.time()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            if use_gpu:
                inputs, labels_phase = data[0].to(device), data[1].type(torch.LongTensor).to(device)
            else:
                inputs, labels_phase = data[0], data[1]

            labels_phase = labels_phase[(sequence_length - 1) :: sequence_length]

            inputs = inputs.view(-1, sequence_length, 3, 224, 224)
            outputs_phase, feats = model.forward(inputs)
            outputs_phase = outputs_phase[sequence_length - 1 :: sequence_length]
            #outputs_phase = outputs_phase.to(torch.float64)

            _, preds_phase = torch.max(outputs_phase.data, 1)
            loss_phase = criterion_phase(outputs_phase, labels_phase)
            center_loss = criterion_center(feats, labels_phase)

            loss = loss_phase + cl_lambda * center_loss
            loss_rec = loss
            loss.backward()
            optimizer.step()
            optimizer_center.step()

            running_loss_phase += loss_rec.data.item()
            train_loss_phase += loss_rec.data.item()

            batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
            train_corrects_phase += batch_corrects_phase
            minibatch_correct_phase += batch_corrects_phase

            if i % 1500 == 1499:
                # ...log the running loss
                batch_iters = (
                    epoch * num_train_all / sequence_length
                    + i * train_batch_size / sequence_length
                )
                writer.add_scalar(
                    "training loss phase",
                    running_loss_phase / (train_batch_size * 500 / sequence_length),
                    batch_iters,
                )
                # ...log the training acc
                writer.add_scalar(
                    "training acc phase",
                    float(minibatch_correct_phase)
                    / (float(train_batch_size) * 500 / sequence_length),
                    batch_iters,
                )
                # ...log the val acc loss

                model.eval()
                criterion_phase = nn.CrossEntropyLoss(size_average=False)
                val_loss_phase = 0.0
                val_corrects_phase = 0.0
                with torch.no_grad():
                    
                    for data in val_loader:
                        if use_gpu:
                            inputs, labels_phase = data[0].to(device), data[1].type(torch.LongTensor).to(device)
                        else:
                            inputs, labels_phase = data[0], data[1]

                        labels_phase = labels_phase[(sequence_length - 1) :: sequence_length]

                        inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                        outputs_phase = model.forward(inputs)
                        outputs_phase = outputs_phase[sequence_length - 1 :: sequence_length]

                        _, preds_phase = torch.max(outputs_phase.data, 1)
                        loss_phase = criterion_phase(outputs_phase, labels_phase)

                        val_loss_phase += loss_phase.data.item()
                        val_corrects_phase += torch.sum(preds_phase == labels_phase.data)
                model.train()

                writer.add_scalar(
                    "validation acc miniBatch phase",
                    float(val_corrects_phase) / float(num_val),
                    batch_iters,
                )
                writer.add_scalar(
                    "validation loss miniBatch phase",
                    float(val_loss_phase) / float(num_val),
                    batch_iters,
                )

                running_loss_phase = 0.0
                minibatch_correct_phase = 0.0

            if (i + 1) * train_batch_size >= num_train_all:
                running_loss_phase = 0.0
                minibatch_correct_phase = 0.0

            batch_progress += 1
            if batch_progress * train_batch_size >= num_train_all:
                percent = 100.0
                print(
                    "Batch progress: %s [%d/%d]"
                    % (str(percent) + "%", num_train_all, num_train_all),
                    end="\n",
                )
            else:
                percent = round(
                    batch_progress * train_batch_size / num_train_all * 100, 2
                )
                print(
                    "Batch progress: %s [%d/%d]"
                    % (
                        str(percent) + "%",
                        batch_progress * train_batch_size,
                        num_train_all,
                    ),
                    end="\r",
                )

        train_elapsed_time = time.time() - train_start_time
        train_accuracy_phase = (
            float(train_corrects_phase) / float(num_train_all) * sequence_length
        )
        train_average_loss_phase = train_loss_phase / num_train_all * sequence_length

        # Sets the module in evaluation mode.
        model.eval()
        val_loss_phase = 0.0
        val_corrects_phase = 0
        val_start_time = time.time()
        val_progress = 0
        val_all_preds_phase = []
        val_all_labels_phase = []

        with torch.no_grad():
            for data in val_loader:
                if use_gpu:
                    inputs, labels_phase = data[0].to(device), data[1].type(torch.LongTensor).to(device)
                else:
                    inputs, labels_phase = data[0], data[1]

                labels_phase = labels_phase[(sequence_length - 1) :: sequence_length]

                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs_phase, _ = model.forward(inputs)
                outputs_phase = outputs_phase[sequence_length - 1 :: sequence_length]

                _, preds_phase = torch.max(outputs_phase.data, 1)
                loss_phase = criterion_phase(outputs_phase, labels_phase)

                val_loss_phase += loss_phase.data.item()

                val_corrects_phase += torch.sum(preds_phase == labels_phase.data)
                # TODO

                for i in range(len(preds_phase)):
                    val_all_preds_phase.append(int(preds_phase.data.cpu()[i]))
                for i in range(len(labels_phase)):
                    val_all_labels_phase.append(int(labels_phase.data.cpu()[i]))

                val_progress += 1
                if val_progress * val_batch_size >= num_val_all:
                    percent = 100.0
                    print(
                        "Val progress: %s [%d/%d]"
                        % (str(percent) + "%", num_val_all, num_val_all),
                        end="\n",
                    )
                else:
                    percent = round(
                        val_progress * val_batch_size / num_val_all * 100, 2
                    )
                    print(
                        "Val progress: %s [%d/%d]"
                        % (
                            str(percent) + "%",
                            val_progress * val_batch_size,
                            num_val_all,
                        ),
                        end="\r",
                    )

        val_elapsed_time = time.time() - val_start_time
        val_accuracy_phase = float(val_corrects_phase) / float(num_val)
        val_average_loss_phase = val_loss_phase / num_val

        val_recall_phase = metrics.recall_score(
            val_all_labels_phase, val_all_preds_phase, average="macro"
        )
        val_precision_phase = metrics.precision_score(
            val_all_labels_phase, val_all_preds_phase, average="macro"
        )
        val_jaccard_phase = metrics.jaccard_score(
            val_all_labels_phase, val_all_preds_phase, average="macro"
        )
        val_precision_each_phase = metrics.precision_score(
            val_all_labels_phase, val_all_preds_phase, average=None
        )
        val_recall_each_phase = metrics.recall_score(
            val_all_labels_phase, val_all_preds_phase, average=None
        )

        writer.add_scalar(
            "validation acc epoch phase", float(val_accuracy_phase), epoch
        )
        writer.add_scalar(
            "validation loss epoch phase", float(val_average_loss_phase), epoch
        )

        logging.info(
            f"epoch:{epoch} train in:{train_elapsed_time//60:.2f}m{train_elapsed_time%60:.2f} train loss(phase):{train_average_loss_phase:.4f} train acc(phase):{train_accuracy_phase:.4f} valid in:{val_elapsed_time//60:.2f}m{val_elapsed_time%60:.2f}s valid loss(phase):{val_average_loss_phase:.4f} valid accu(phase):{val_accuracy_phase:.4f}"
        )

        logging.info(f"val_precision_each_phase:{val_precision_each_phase}")
        logging.info(f"val_recall_each_phase:{val_recall_each_phase}")
        logging.info(f"val_precision_phase: {val_precision_phase}")
        logging.info(f"val_recall_phase: {val_recall_phase}")
        logging.info(f"val_jaccard_phase: {val_jaccard_phase}")

        exp_lr_scheduler.step(val_average_loss_phase)

        if val_accuracy_phase > best_val_accuracy_phase:
            best_val_accuracy_phase = val_accuracy_phase
            correspond_train_acc_phase = train_accuracy_phase
            best_model_wts = copy.deepcopy(model.module.state_dict())
            best_epoch = epoch

            # Best stats
            best_tr_loss = train_average_loss_phase
            best_tr_acc = train_accuracy_phase
            best_v_loss = val_average_loss_phase
            best_v_acc = val_accuracy_phase
            best_v_pr = val_precision_phase
            best_v_re = val_recall_phase
            best_v_ji = val_jaccard_phase
            best_v_pr_each = val_precision_each_phase
            best_v_re_each = val_recall_each_phase

            torch.save(
                best_model_wts,
                exp_path + "/checkpoints/spatial_extractor_best.pth",
            )
        if val_accuracy_phase == best_val_accuracy_phase:
            if train_accuracy_phase > correspond_train_acc_phase:
                correspond_train_acc_phase = train_accuracy_phase
                best_model_wts = copy.deepcopy(model.module.state_dict())
                best_epoch = epoch

                # Best stats
                best_tr_loss = train_average_loss_phase
                best_tr_acc = train_accuracy_phase
                best_v_loss = val_average_loss_phase
                best_v_acc = val_accuracy_phase
                best_v_pr = val_precision_phase
                best_v_re = val_recall_phase
                best_v_ji = val_jaccard_phase
                best_v_pr_each = val_precision_each_phase
                best_v_re_each = val_recall_each_phase

            torch.save(
                best_model_wts,
                exp_path + "/checkpoints/spatial_extractor_best.pth",
            )

        logging.info(f"best_epoch: {best_epoch}")

    df_new_row = pd.DataFrame([[fold, "spatial_extractor", best_epoch,
                                best_tr_loss, best_tr_acc, best_v_loss, best_v_acc, 0,
                                best_v_pr, best_v_re, best_v_ji,
                                best_v_pr_each, best_v_re_each]],
                                columns=['fold', 'module', 'best_epoch',
                                            'best_train_loss', 'best_train_acc', 'best_val_loss', 'best_val_acc', 'best_val_acc_vid',
                                            'best_val_precision', 'best_val_recall', 'best_val_jaccard',
                                            'best_val_precision_each', 'best_val_recall_each'])
    df_res = pd.read_csv(exp_path[:exp_path.index('fold')-1] + "/best_res.csv")
    df_res = pd.concat([df_res, df_new_row], ignore_index=True)
    df_res.to_csv(exp_path[:exp_path.index('fold')-1] + "/best_res.csv", index=False)

    logging.info(f"best_epoch: {str(best_epoch)}")
    logging.info(f"best stats:")
    logging.info(f"best_train_loss:{best_tr_loss:.4f} best_train_acc:{best_tr_acc:.4f} best_val_loss:{best_v_loss:.4f} best_val_acc:{best_v_acc:.4f}")
    logging.info(f"best_val_precision:{best_v_pr:.4f} best_val_recall:{best_v_re:.4f} best_val_jaccard:{best_v_ji:.4f}")
    logging.info(f"best_val_precision_each:{best_v_pr_each} best_val_recall_each:{best_v_re_each}")

    logging.info("Done")
    print()


def train_temporal_extractor(config, use_gpu, exp_path, fold):
    num_gpu = torch.cuda.device_count()
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    
    dataset = config.general["dataset"]
    label_path = config.dataset_dirs[dataset]["label_path"]
    epochs = config.general["epochs"]
    adam_lr = config.temporal_extractor["adam_lr"]

    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S :")
    fh = logging.FileHandler(exp_path + "/logs/ltc.txt") 
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info(f"number of gpu        :{num_gpu}")
    logging.info(f"dataset              :{dataset}")
    logging.info(f"label directory path :{label_path}")
    logging.info(f"number of epochs     :{epochs}")
    logging.info(f"adam learning rate   :{adam_lr}")

    train_labels, train_num_each, train_start_vidx, \
        val_labels, val_num_each, val_start_vidx = get_data_ltc(label_path, fold)

    # Get saved features
    with open(exp_path + "/features/features_train.pkl", 'rb') as f:
        feats_train = pickle.load(f)
    with open(exp_path + "/features/features_val.pkl", 'rb') as f:
        feats_val = pickle.load(f)

    logging.info('load completed')
    logging.info(f'feats_train shape: {feats_train.shape}')
    logging.info(f'feats_val shape: {feats_val.shape}')

    seed = 1
    logging.info(f'Random Seed: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    df_train = pd.read_csv(label_path + f"{fold}train.csv")
    df_val = pd.read_csv(label_path + f"{fold}val.csv")

    labels_train = df_train["class"].tolist()
    labels_val = df_val["class"].tolist()
    labels_all = labels_train + labels_val

    weights_train = class_weight.compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2, 3, 4]), y=labels_all) #####

    criterion_phase = nn.CrossEntropyLoss(weight=torch.from_numpy(weights_train).float().to(device))

    model = LTC()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=adam_lr)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy_phase = 0.0
    correspond_train_acc_phase = 0.0
    best_epoch = 0

    # Record best stats
    best_tr_loss = 0.0
    best_tr_acc = 0.0
    best_v_loss = 0.0
    best_v_acc = 0.0
    best_v_acc_vid = 0.0
    best_v_pr = 0.0
    best_v_re = 0.0
    best_v_ji = 0.0
    best_v_pr_each = 0.0
    best_v_re_each = 0.0

    train_start_idx = [x for x in range(len(train_num_each))]
    val_start_idx = [x for x in range(len(val_num_each))]
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        model.train()
        train_loss_phase = 0.0
        train_corrects_phase = 0
        running_loss_phase = 0.0
        minibatch_correct_phase = 0.0
        train_start_time = time.time()
        for i in train_start_idx:
            optimizer.zero_grad()
            labels_phase = []
            for j in range(train_start_vidx[i], train_start_vidx[i]+train_num_each[i]):
                labels_phase.append(train_labels[j])
            labels_phase = torch.LongTensor(labels_phase)
            if use_gpu:
                labels_phase = labels_phase.to(device)
            else:
                labels_phase = labels_phase
            long_feature = get_long_feature(start_index=train_start_vidx[i],
                                            feats=feats_train, feat_length=train_num_each[i])

            long_feature = (torch.Tensor(long_feature)).to(device)
            video_fe = long_feature.transpose(2, 1)

            y_classes = model.forward(video_fe)
            stages = y_classes.shape[0]
            clc_loss = 0
            for j in range(stages):
                p_classes = y_classes[j].squeeze().transpose(1, 0)
                ce_loss = criterion_phase(p_classes, labels_phase)
                clc_loss += ce_loss
            clc_loss = clc_loss / (stages * 1.0)

            _, preds_phase = torch.max(y_classes[stages-1].squeeze().transpose(1, 0).data, 1)

            loss = clc_loss
            loss.backward()
            optimizer.step()

            running_loss_phase += clc_loss.data.item()
            train_loss_phase += clc_loss.data.item()

            batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
            train_corrects_phase += batch_corrects_phase
            minibatch_correct_phase += batch_corrects_phase

        train_elapsed_time = time.time() - train_start_time
        train_accuracy_phase = float(train_corrects_phase) / len(train_labels)
        train_average_loss_phase = train_loss_phase

        # Sets the module in evaluation mode.
        model.eval()
        val_loss_phase = 0.0
        val_corrects_phase = 0
        val_start_time = time.time()
        val_all_preds_phase = []
        val_all_labels_phase = []
        val_acc_each_video = []

        with torch.no_grad():
            for i in val_start_idx:
                labels_phase = []
                for j in range(val_start_vidx[i], val_start_vidx[i] + val_num_each[i]):
                    labels_phase.append(val_labels[j])
                labels_phase = torch.LongTensor(labels_phase)
                if use_gpu:
                    labels_phase = labels_phase.to(device)
                else:
                    labels_phase = labels_phase

                long_feature = get_long_feature(start_index=val_start_vidx[i],
                                                feats=feats_val, feat_length=val_num_each[i])

                long_feature = (torch.Tensor(long_feature)).to(device)
                video_fe = long_feature.transpose(2, 1)

                y_classes = model.forward(video_fe)
                stages = y_classes.shape[0]
                clc_loss = 0
                for j in range(stages):
                    p_classes = y_classes[j].squeeze().transpose(1, 0)
                    ce_loss = criterion_phase(p_classes, labels_phase)
                    clc_loss += ce_loss
                clc_loss = clc_loss / (stages * 1.0)

                _, preds_phase = torch.max(y_classes[stages - 1].squeeze().transpose(1, 0).data, 1)
                p_classes = y_classes[-1].squeeze().transpose(1, 0)
                loss_phase = criterion_phase(p_classes, labels_phase)

                val_loss_phase += loss_phase.data.item()

                val_corrects_phase += torch.sum(preds_phase == labels_phase.data)
                val_acc_each_video.append(float(torch.sum(preds_phase == labels_phase.data))/val_num_each[i])

                for j in range(len(preds_phase)):
                    val_all_preds_phase.append(int(preds_phase.data.cpu()[j]))
                for j in range(len(labels_phase)):
                    val_all_labels_phase.append(int(labels_phase.data.cpu()[j]))


        val_elapsed_time = time.time() - val_start_time
        val_accuracy_phase = float(val_corrects_phase) / len(val_labels)
        val_acc_video = np.mean(val_acc_each_video)
        val_average_loss_phase = val_loss_phase

        val_recall_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average='macro')
        val_precision_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average='macro')
        val_jaccard_phase = metrics.jaccard_score(val_all_labels_phase, val_all_preds_phase, average='macro')
        val_precision_each_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average=None)
        val_recall_each_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average=None)

        logging.info(f"epoch:{epoch} train in:{train_elapsed_time//60:.2f}m{train_elapsed_time%60:.2f} train loss(phase):{train_average_loss_phase:.4f} train acc(phase):{train_accuracy_phase:.4f} valid in:{val_elapsed_time//60:.2f}m{val_elapsed_time%60:.2f}s valid loss(phase):{val_average_loss_phase:.4f} valid accu(phase):{val_accuracy_phase:.4f} valid accu(video):{val_acc_video:.4f}")
        logging.info(f"val_precision_each_phase:{val_precision_each_phase}")
        logging.info(f"val_recall_each_phase:{val_recall_each_phase}")
        logging.info(f"val_precision_phase: {val_precision_phase}")
        logging.info(f"val_recall_phase: {val_recall_phase}")
        logging.info(f"val_jaccard_phase: {val_jaccard_phase}")

        if val_accuracy_phase > best_val_accuracy_phase:
            best_val_accuracy_phase = val_accuracy_phase
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch

            # Best stats
            best_tr_loss = train_average_loss_phase
            best_tr_acc = train_accuracy_phase
            best_v_loss = val_average_loss_phase
            best_v_acc = val_accuracy_phase
            best_v_acc_vid = val_acc_video
            best_v_pr = val_precision_phase
            best_v_re = val_recall_phase
            best_v_ji = val_jaccard_phase
            best_v_pr_each = val_precision_each_phase
            best_v_re_each = val_recall_each_phase
        
            torch.save(best_model_wts, exp_path + "/checkpoints/temporal_extractor_best.pth")
        
        logging.info(f"best_epoch: {best_epoch}")
        
    df_new_row = pd.DataFrame([[fold, "temporal_extractor", best_epoch,
                                best_tr_loss, best_tr_acc, best_v_loss, best_v_acc, best_v_acc_vid,
                                best_v_pr, best_v_re, best_v_ji,
                                best_v_pr_each, best_v_re_each]],
                                columns=['fold', 'module', 'best_epoch',
                                            'best_train_loss', 'best_train_acc', 'best_val_loss', 'best_val_acc', 'best_val_acc_vid',
                                            'best_val_precision', 'best_val_recall', 'best_val_jaccard',
                                            'best_val_precision_each', 'best_val_recall_each'])
    df_res = pd.read_csv(exp_path[:exp_path.index('fold')-1] + "/best_res.csv")
    df_res = pd.concat([df_res, df_new_row], ignore_index=True)
    df_res.to_csv(exp_path[:exp_path.index('fold')-1] + "/best_res.csv", index=False)
    
    logging.info(f'best_epoch {str(best_epoch)}')
    logging.info(f"best stats:")
    logging.info(f"best_train_loss:{best_tr_loss:.4f} best_train_acc:{best_tr_acc:.4f} best_val_loss:{best_v_loss:.4f} best_val_acc:{best_v_acc:.4f} best_val_acc_vid:{best_v_acc_vid:.4f}")
    logging.info(f"best_val_precision:{best_v_pr:.4f} best_val_recall:{best_v_re:.4f} best_val_jaccard:{best_v_ji:.4f}")
    logging.info(f"best_val_precision_each:{best_v_pr_each} best_val_recall_each:{best_v_re_each}")

    logging.info("Done")
    print()


# if __name__ == "__main__":
#     train_spatial_extractor()
#     train_temporal_extractor()
