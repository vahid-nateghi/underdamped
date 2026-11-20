import os
import sys
sys.path.append("../")  # add parent dir to path
import numpy as np
import pandas as pd
import h5py

if __name__ == "__main__":
    # hparams
    n = 1000_000  # nr samples
    burnin = 1000  # burnin for sampling
    stride = 1  # stride for sampling
    betas = [0.25, 0.5, 0.75, 1., 1.5, 2.] # inverse temps
    # empy df hierarchical indexing to fit (4, m) shape of data
    cols = pd.MultiIndex.from_product([[f"{beta:.2f}" for beta in betas], 
                                       ["x", "y", "p_x", "p_y"]], names=["beta", "coordinate"])
    # create empty df
    df = pd.DataFrame(columns=cols, index=range(n))
    for i, beta in enumerate(betas):
        with h5py.File(input.data_path+f'/lemon_data_rs_beta_{beta:.2f}.h5', 'r') as f:
            lemon_rs_data = f['x_fs'][()]
        df.iloc[:, i*4:i*4+4] = lemon_rs_data.T

    # save
    if not os.path.exists(".dataset"):
        os.makedirs(".dataset")

    df.to_csv(".dataset/lemon_slice.csv")

    # create train/val/test splits
    ratios = (0.8, 0.1, 0.1)

    train_df = pd.DataFrame(columns=df.columns, index=range(int(len(df)*ratios[0])))
    val_df = pd.DataFrame(columns=df.columns, index=range(int(len(df)*ratios[1])))
    test_df = pd.DataFrame(columns=df.columns, index=range(int(len(df)*ratios[2])))

    # fill train/val/test splits with sampled (no replacement) indices
    train_indices = np.random.choice(df.index, size=int(len(df)*ratios[0]), replace=False)
    remaining_indices = np.setdiff1d(df.index, train_indices)

    val_indices = np.random.choice(remaining_indices, size=int(len(df)*ratios[1]), replace=False)
    remaining_indices = np.setdiff1d(remaining_indices, val_indices)

    test_indices = remaining_indices

    # fill train/val/test splits with sampled (no replacement) indices
    train_df.iloc[:, :] = df.iloc[train_indices, :]
    val_df.iloc[:, :] = df.iloc[val_indices, :]
    test_df.iloc[:, :] = df.iloc[test_indices, :]

    # save train/val/test splits to csv
    train_df.to_csv(".dataset/train.csv")
    val_df.to_csv(".dataset/val.csv")
    test_df.to_csv(".dataset/test.csv")
