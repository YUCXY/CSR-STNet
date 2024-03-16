import os
import yaml
import subprocess
import argparse
from time import strftime, localtime
import pandas as pd

from utils.train_loops import train_spatial_extractor, train_temporal_extractor
from save_features import save_features
from utils.utils import make_exp_dirs



def main(config, use_gpu):

    save_dir = config.general["save_dir"]
    exp = strftime("%m%d%H%M%S", localtime())
    os.makedirs(f"./{save_dir}/{exp}")

    df_res = pd.DataFrame({'fold':[], 'module':[], 'best_epoch':[],
                           'best_train_loss':[], 'best_train_acc':[], 'best_val_loss':[], 'best_val_acc':[], 'best_val_acc_vid':[],
                           'best_val_precision':[], 'best_val_recall':[], 'best_val_jaccard':[],
                           'best_val_precision_each':[], 'best_val_recall_each':[]})
    df_res.to_csv(f"./{save_dir}/{exp}/best_res.csv", index=False)

    for fold in range(0, 5):
        exp_path = f"./{save_dir}/{exp}/fold{fold}"
        make_exp_dirs(exp_path)

        train_spatial_extractor(config, use_gpu, exp_path, fold)
        save_features(config, use_gpu, exp_path, fold)
        train_temporal_extractor(config, use_gpu, exp_path, fold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CSR-STNet")
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config = argparse.Namespace(**config)

    main(config, args.use_gpu)