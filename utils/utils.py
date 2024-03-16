import os
import pandas as pd
import numpy as np
from torch.utils.data import Sampler
from torchvision import transforms
from .dataset import ColoDataset
from .augmentations import RandomCrop, ColourJitter, RandomHorizontalFlip, RandomRotation



def make_exp_dirs(exp_path):
    ckpt_path = exp_path + "/checkpoints"
    feat_path = exp_path + "/features"
    log_path = exp_path + "/logs"
    run_path = exp_path + "/runs"

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.exists(feat_path):
        os.makedirs(feat_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(run_path):
        os.makedirs(run_path)


def get_start_idx(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx


def get_num_each(df):
    count_dict = {}
    for i in range(df.shape[0]):
        name = df.iloc[i]['img']
        name = name[:name.index('_frame')]
        if name in count_dict.keys(): count_dict[name] += 1
        else: count_dict[name] = 1
    count_vals = count_dict.values()
    return list(count_vals)


def get_features(start_index_list, dict_start_idx_feats, features, feature_len):
    long_feature = []
    for j in range(len(start_index_list)):
        long_feature_each = []

        last_feat_index_no_empty = dict_start_idx_feats[int(start_index_list[j])]

        for k in range(feature_len):
            LFB_index = (start_index_list[j] - k - 1)
            if int(LFB_index) in dict_start_idx_feats:
                LFB_index = dict_start_idx_feats[int(LFB_index)]
                long_feature_each.append(features[LFB_index])
                last_feat_index_no_empty = LFB_index
            else:
                long_feature_each.append(features[last_feat_index_no_empty])

        long_feature.append(long_feature_each)
    return long_feature


def get_long_feature(start_index, feats, feat_length):
    long_feature = []
    long_feature_each = []
    for k in range(feat_length):
        feat_index = (start_index + k)
        feat_index = int(feat_index)
        long_feature_each.append(feats[feat_index])
    long_feature.append(long_feature_each)
    return long_feature


def get_data_ltc(label_path, fold):
    df_train = pd.read_csv(label_path + f'{fold}train.csv')
    df_val = pd.read_csv(label_path + f'{fold}val.csv')

    train_labels = df_train["class"].tolist()
    val_labels = df_val["class"].tolist()

    train_num_each = get_num_each(df_train)
    val_num_each = get_num_each(df_val)

    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)

    train_start_vidx = []
    count = 0
    for i in range(len(train_num_each)):
        train_start_vidx.append(count)
        count += train_num_each[i]

    val_start_vidx = []
    count = 0
    for i in range(len(val_num_each)):
        val_start_vidx.append(count)
        count += val_num_each[i]

    return train_labels, train_num_each, train_start_vidx, val_labels, val_num_each, val_start_vidx


def get_data(img_path, label_path, fold, sequence_length, get_feats=False):
    train_transforms = transforms.Compose(
        [
            transforms.Resize((250, 250)),
            RandomCrop(224, sequence_length),
            ColourJitter(sequence_length, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            RandomHorizontalFlip(sequence_length),
            RandomRotation(5, sequence_length),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.41757566, 0.26098573, 0.25888634],
                [0.21938758, 0.1983, 0.19342837],
            ),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((250, 250)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.41757566, 0.26098573, 0.25888634],
                [0.21938758, 0.1983, 0.19342837],
            ),
        ]
    )

    df_train = pd.read_csv(label_path + f"{fold}train.csv")
    df_val = pd.read_csv(label_path + f"{fold}val.csv")

    train_paths = [img_path + i + ".jpg" for i in df_train["img"].tolist()]
    val_paths = [img_path + i + ".jpg" for i in df_val["img"].tolist()]

    train_labels = df_train["class"].tolist()
    val_labels = df_val["class"].tolist()

    train_num_each = get_num_each(df_train)
    val_num_each = get_num_each(df_val)

    train_dataset = ColoDataset(train_paths, train_labels, train_transforms)
    train_dataset_feats = ColoDataset(train_paths, train_labels, test_transforms)
    val_dataset = ColoDataset(val_paths, val_labels, test_transforms)

    if get_feats:
        return (train_dataset, train_dataset_feats), train_num_each, val_dataset, val_num_each
    else:
        return (train_dataset, train_num_each, val_dataset, val_num_each)


class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)