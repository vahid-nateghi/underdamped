import numpy as np
import pandas as pd

import torch
from torch.utils import data


LEMON_SLICE_STD = 0.78139776


class LemonSliceDataset(data.Dataset):

    def __init__(self, n_samples, betas: list, traj_path: str=".dataset", seed: int=0, device: str="cpu", dtype: torch.dtype=torch.float32, split: str="train"):
        """## Init

        ### Args:
            - `n_samples (_type_)`: Number of samples to take from the dataset.
            - `betas (list, optional)`: List of inverse temperatures.
            - `traj_path (str, optional)`: Path to the trajectory file. Defaults to "dataset".
            - `seed (int, optional)`: Random seed for reproducibility. Defaults to 0.
            - `device (str, optional)`: Device to load the data on. Defaults to "cpu".
            - `dtype (torch.dtype, optional)`: Data type of the tensor. Defaults to torch.float32.
            - `split (str, optional)`: Split of the dataset to load. Defaults to "train".
        """

        assert split in {"train", "val", "test"}, f"split must be one of ['train', 'val', 'test'], got {split}"

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.data = self.load(
            n_samples=n_samples,
            betas=betas,
            path=traj_path,
            device=device,
            dtype=dtype,
            split=split,
        )

        super().__init__()
    
    def __len__(self) -> int:
        """## Length of the dataset.

        ### Returns:
            - `int`: Length of the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple:
        """## Get item from the dataset.

        ### Args:
            - `idx (int)`: Index of the sample to retrieve.

        ### Returns:
            - `tuple`: Tuple containing the sample position and corresponding beta.
        """

        item = self.data[idx]

        pos = item[:2]  # positions + momentum
        pos = pos/LEMON_SLICE_STD

        beta = item[-1].unsqueeze(-1)  # inv temp
        return (pos, beta)
    
    def calc_std(self, betas: list, path: str=".dataset"):
        """## Calculate the standard deviation of the dataset.

        ### Args:
            - `path (str, optional)`: Path to the dataset. Defaults to ".dataset".

        ### Returns:
            - `float`: Standard deviation of the dataset.
        """

        betas = [f"{beta:.2f}" for beta in betas]
        df = pd.read_csv(f"{path}/train.csv", header=[0, 1])  # always calc based on train set to avoid data leakage
        posx = df[betas].loc[:, pd.IndexSlice[:, "x"]].values
        posy = df[betas].loc[:, pd.IndexSlice[:, "y"]].values

        pos = np.concatenate([posx, posy], axis=-1).flatten()

        px = df[betas].loc[:, pd.IndexSlice[:, "p_x"]].values
        py = df[betas].loc[:, pd.IndexSlice[:, "p_y"]].values

        p = np.concatenate([px, py], axis=-1).flatten()

        std_pos = np.std(pos)
        std_mom = np.std(p)
        return std_pos, std_mom

    @staticmethod
    def load(n_samples, betas: list, path: str=".dataset", device: str="cpu", dtype: torch.dtype=torch.float32, split: str="train"):
        betas = [f"{beta:.2f}" for beta in betas]
        df = pd.read_csv(f"{path}/{split}.csv", header=[0, 1])

        # get subcol x for each beta
        xs = df[betas].loc[:, pd.IndexSlice[:, "x"]].values
        ys = df[betas].loc[:, pd.IndexSlice[:, "y"]].values

        pxs = df[betas].loc[:, pd.IndexSlice[:, "p_x"]].values
        pys = df[betas].loc[:, pd.IndexSlice[:, "p_y"]].values

        n_samples_per_beta, _ = xs.shape

        xs = np.concatenate(xs.T, axis=0)
        ys = np.concatenate(ys.T, axis=0)
        pxs = np.concatenate(pxs.T, axis=0)
        pys = np.concatenate(pys.T, axis=0)

        betas = np.concatenate([[float(beta)] * n_samples_per_beta for beta in betas], axis=0)

        data = np.concatenate([xs[:, np.newaxis], ys[:, np.newaxis], pxs[:, np.newaxis], pys[:, np.newaxis], betas[:, np.newaxis]], axis=1)
        
        assert n_samples <= len(data), f"n_samples ({n_samples}) is greater than the number of samples in the dataset ({len(data)})"

        # convert to torch tensor
        data = torch.tensor(data, dtype=dtype, device=device)
        return data


class StandardGaussianDataset(data.Dataset):

    def __init__(self, n_samples, beta=None, seed: int=0, device: str="cpu", dtype: torch.dtype=torch.float32):
        """## Init

        ### Args:
            - `n_samples (_type_)`: Number of samples to take from the dataset.
            - `betas (list, optional)`: List of inverse temperatures.
            - `traj_filename (str, optional)`: Name of the trajectory file. Defaults to "lemon_slice.csv".
            - `traj_path (str, optional)`: Path to the trajectory file. Defaults to "dataset".
            - `seed (int, optional)`: Random seed for reproducibility. Defaults to 0.
        """

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.include_beta = beta is not None

        self.data = self.load(
            n_samples=n_samples,
            beta=beta,
            device=device,
            dtype=dtype,
        )

        super().__init__()
    
    def __len__(self) -> int:
        """## Length of the dataset.

        ### Returns:
            - `int`: Length of the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple:
        """## Get item from the dataset.

        ### Args:
            - `idx (int)`: Index of the sample to retrieve.

        ### Returns:
            - `tuple`: Tuple containing the sample position and corresponding beta.
        """

        item = self.data[idx]

        pos = torch.randn_like(item[:2])
        if self.include_beta:
            beta = item[-1].unsqueeze(-1)
        else:
            beta = torch.tensor(0.0, device=pos.device, dtype=pos.dtype).unsqueeze(-1)  # dummy

        return (pos, beta)

    def load(self, n_samples, beta, device: str="cpu", dtype: torch.dtype=torch.float32):
        data = torch.ones((n_samples, 4), device=device, dtype=dtype)

        if self.include_beta:
            betas = torch.tensor(beta, device=device, dtype=dtype).unsqueeze(0).expand(n_samples, -1)
            data = torch.cat([data, betas], dim=-1)  
        return data
    

