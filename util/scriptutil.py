import os
import json
import argparse

import numpy as np

import torch
from torch.utils.data import random_split, DataLoader, Dataset

from thermo import interpolants, losses
from data import datasets


def get_interpolant(config: argparse.Namespace):
    if config.interpolation_strategy == "one-sided":
        return interpolants.OneSidedLinearInterpolant()
    elif config.interpolation_strategy == "two-sided":
        return interpolants.LinearInterpolant(a=config.gamma_a)
    else:
        raise ValueError(f"Unknown interpolation strategy: {config.interpolation_strategy}")


def get_loss(interpolant, config: argparse.Namespace):
    if config.interpolation_strategy == "one-sided":
        return losses.OneSidedVelocityLoss(interpolant=interpolant)
    elif config.interpolation_strategy == "two-sided":
        return losses.TwoSidedVelocityLoss(interpolant=interpolant)
    else:
        raise ValueError(f"Unknown interpolation strategy: {config.interpolation_strategy}")


def get_loaders(split, config: argparse.Namespace, device, dtype=torch.float32, epoch=0) -> DataLoader:
    assert split in ["train", "val", "test"], f"Unknown split: {split}"

    if split in ["train", "val"]:
        # --- init base loader ---
        if config.interpolation_strategy == "one-sided":
            base_data = datasets.StandardGaussianDataset(
                n_samples=config.n_train_samples if split == "train" else config.n_val_samples,
                beta=None,
                device=device,
                dtype=dtype,
            )

        elif config.interpolation_strategy == "two-sided":
            base_data = datasets.LemonSliceDataset(
                n_samples=config.n_train_samples if split == "train" else config.n_val_samples,
                betas=config.beta_train,
                traj_path=config.traj_path,
                device=device,
                dtype=dtype,
                split=split,
            )
            base_data = torch.utils.data.Subset(base_data, np.random.choice(len(base_data), config.n_train_samples if split == "train" else config.n_val_samples, replace=False))

        else:
            raise ValueError(f"Unknown interpolation strategy: {config.interpolation_strategy}")

        base_loader = DataLoader(
            dataset=base_data,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(config.seed + epoch)
            )
        
        # --- init target loader ---
        target_data = datasets.LemonSliceDataset(
                n_samples=config.n_train_samples if split == "train" else config.n_val_samples,
                betas=config.beta_train,
                traj_path=config.traj_path,
                device=device,
                dtype=dtype,
                split=split,
            )
        
        target_data = torch.utils.data.Subset(target_data, np.random.choice(len(target_data), config.n_train_samples if split == "train" else config.n_val_samples, replace=False))

        target_loader = DataLoader(
            dataset=target_data,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(config.seed + epoch + 1)  # avoid overlap with base_loader when two-sided
        )

        assert len(base_loader) == len(target_loader), f"Base and target loaders must have the same length, currently {len(base_loader)} (base) and {len(target_loader)} (target)"
        return base_loader, target_loader

    elif split == "test":
        # --- init base loader (no target loader for test split) ---
        if config.interpolation_strategy == "one-sided":
            base_data = datasets.StandardGaussianDataset(
                n_samples=config.n_test_samples,
                beta=config.beta_test[0],
                device=device,
                dtype=dtype,
            )

        elif config.interpolation_strategy == "two-sided":
            base_data = datasets.LemonSliceDataset(
                n_samples=config.n_test_samples,
                betas=[config.beta_test[0]],
                traj_path=config.traj_path,
                device=device,
                dtype=dtype,
                split=split,
            )

        else:
            raise ValueError(f"Unknown interpolation strategy: {config.interpolation_strategy}")
        
        base_data = torch.utils.data.Subset(base_data, np.random.choice(len(base_data), config.n_test_samples, replace=False))
        base_loader = DataLoader(
            base_data,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True
        )

        return base_loader


def load_config(path: str, filename: str) -> argparse.Namespace:
    settings = json.load(open(os.path.join(path, filename), 'r'))

    parser = argparse.ArgumentParser()
    
    for key, value in settings.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    return parser.parse_args()


def save_config(config: argparse.Namespace):
    if not os.path.exists(os.path.join(config.model_save_path, config.model_save_name)):
        os.makedirs(os.path.join(config.model_save_path, config.model_save_name))
    
    with open(os.path.join(config.model_save_path, config.model_save_name, "settings.json"), 'w') as f:
        json.dump(vars(config), f, indent=4)
    f.close()


def add_to_json(json_path, data):
    if not os.path.exists(json_path):
        with open(json_path, 'w') as f:
            json.dump({}, f)
        f.close()

    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    json_data.update(data)
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    f.close()
