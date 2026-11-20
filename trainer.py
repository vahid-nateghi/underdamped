import os
import tqdm
import torch
import wandb
import numpy as np

import util.scriptutil as scriptutil
from thermo.models.simple import VelocityModel

import sampler
import util.metrics as metrics
from data import datasets

import warnings
warnings.filterwarnings("ignore")


def train(config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    if config.use_wandb:
        wandb.init(project=config.project_name, config=config)

    # models
    if config.interpolation_strategy == "one-sided":
        n_betas = 1
    elif config.interpolation_strategy == "two-sided":
        n_betas = 2
    else:
        raise ValueError(f"Unknown interpolation strategy: {config.interpolation_strategy}")

    b = VelocityModel(
        in_dim=2,
        out_dim=2, 
        hidden_size=config.hidden_size, 
        num_layers=config.num_layers,
        n_betas=n_betas,
        beta_length=config.beta_length,
    )
    b.to(device)

    # interpolant
    interpolant = scriptutil.get_interpolant(config)

    # define loss etc.
    loss_fn = scriptutil.get_loss(interpolant, config)
    optim = torch.optim.Adam(b.parameters(), lr=config.lr, weight_decay=config.wd)
    lr_s = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=10, verbose=True)

    # main tqdm loop
    for epoch in tqdm.tqdm(range(config.epochs), desc='Training progress (epochs)', leave=True, total=config.epochs):
        epoch_train_losses = []
        epoch_val_losses = []

        train_base, train_target = scriptutil.get_loaders("train", config, device, epoch=epoch)
        val_base, val_target = scriptutil.get_loaders("val", config, device, epoch=epoch)

        # train
        b.train()
        for (pos0, beta0), (pos1, beta1) in tqdm.tqdm(zip(train_base, train_target), desc='Training dataloader progress', leave=False, total=len(train_base)):
            optim.zero_grad()
            loss = loss_fn(b, pos0, pos1, beta0, beta1)

            # "safe" backprop since stochastic interpolants can be prone to NaN
            if torch.isnan(loss).any():
                if config.use_wandb:
                    wandb.log({"NaN log": 1}, step=epoch)
                else:
                    print("NaN loss")
                continue

            loss.backward()

            optim.step()

            epoch_train_losses.append(loss.detach())

        epoch_train_loss = torch.stack(epoch_train_losses).mean()

        # eval
        b.eval()
        for (pos0, beta0), (pos1, beta1) in tqdm.tqdm(zip(val_base, val_target), desc='Validation dataloader progress', leave=False, total=len(val_base)):
            # compute the loss
            loss = loss_fn(b, pos0, pos1, beta0, beta1)
            epoch_val_losses.append(loss.detach())

        if config.log_metrics:
            # create new loader with config.n_test_samples from val_base at beta_test temperatures
            if config.interpolation_strategy == "one-sided":
                val_data = datasets.StandardGaussianDataset(
                    n_samples=config.n_test_samples,
                    beta=None,
                    device=device,
                    dtype=torch.float32,
                )
            elif config.interpolation_strategy == "two-sided":
                val_data = datasets.LemonSliceDataset(
                    n_samples=config.n_test_samples,
                    betas=[config.beta_test[0]],
                    traj_path=config.traj_path,
                    device=device,
                    dtype=torch.float32,
                    split="val",
                )
            else:
                raise ValueError(f"Unknown interpolation strategy: {config.interpolation_strategy}")
            
            val_data = torch.utils.data.Subset(val_data, np.random.choice(len(val_data), config.n_test_samples, replace=False))
            val_base_sample = torch.utils.data.DataLoader(
                dataset=val_data,
                batch_size=config.batch_size,
                shuffle=False,
                drop_last=True,
            )
        
            # compute metrics
            samples, _ = sampler.sample(
                b=b,
                loader=val_base_sample,
                config=config,
                n_samples=config.n_val_samples,
                device=device,
            )

            samples = torch.tensor(samples, device=device, dtype=torch.float32)[-1, :, :2]

            reference_samples = datasets.LemonSliceDataset.load(
                n_samples=config.n_val_samples,
                betas=[config.beta_test[-1]],
                path=config.traj_path,
                device=device,
                dtype=torch.float32,
                split="val",
            )[:, :2]

            variational_distance = metrics.variational_distance(samples, reference_samples)
        

        epoch_val_loss = torch.stack(epoch_val_losses).mean()
        lr_s.step(epoch_val_loss)
        
        if config.use_wandb:
            if config.log_metrics:
                wandb.log({"train_loss": epoch_train_loss, "val_loss": epoch_val_loss, "variational_distance": variational_distance}, step=epoch)
            else:
                wandb.log({"train_loss": epoch_train_loss, "val_loss": epoch_val_loss}, step=epoch)

        else:
            if config.log_metrics:
                print(f"Epoch: {epoch:03d} --- Loss: {epoch_train_loss:.8f} --- Val Loss: {epoch_val_loss:.8f} --- Variational Distance: {variational_distance:.8f}")
            else:
                print(f"Epoch: {epoch:03d} --- Loss: {epoch_train_loss:.8f} --- Val Loss: {epoch_val_loss:.8f}")

        # save models
        if not os.path.exists(os.path.join(config.model_save_path, config.model_save_name)):
            os.makedirs(os.path.join(config.model_save_path, config.model_save_name))
        
        torch.save(b, os.path.join(config.model_save_path, config.model_save_name, f"epoch_{epoch}.pt"))
    
    if config.use_wandb:
        wandb.finish()


def main(config):
    torch.manual_seed(config.seed)

    # save run settings
    scriptutil.save_config(config)

    # start training
    train(config)

if __name__ == "__main__":
    config = scriptutil.load_config("", "settings_ti.json")  # edit config file to edit run
    main(config)
 