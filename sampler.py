import os
import tqdm
import numpy as np
import torch

import thermo.integrators as integrators
import util.scriptutil as scriptutil
import util.weights as weights

import data.datasets as datasets


def sample(config, b, loader, n_samples=None, device="cpu"):
    if n_samples is None:
        n_samples = config.n_test_samples

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    assert len(config.beta_test) == 2, "Only sampling at single \beta is supported"

    integrator = integrators.StandardIntegrator(
        b=b,
        method=config.solver_type,
        n_step=config.n_step,
        return_dlogp=config.return_dlogp,
    )

    samples = []
    dlogps = []

    b.eval()
    for i, (pos0, beta0) in tqdm.tqdm(enumerate(loader), desc='Sampling progress', leave=False, total=len(loader)):
        pos0, beta0 = pos0.to(device), beta0.to(device)
        beta1 = torch.ones_like(beta0)*config.beta_test[1] if config.interpolation_strategy == "two-sided" else None

        pos1_sample, dlogp = integrator.rollout(pos0, beta0, beta1)
        if not config.return_dlogp:
            pos1_sample, = pos1_sample

        samples.append(pos1_sample.detach().cpu().numpy())

        if config.return_dlogp:
            dlogps.append(dlogp.detach().cpu().numpy())

    if not os.path.exists(config.data_save_path):
        os.makedirs(config.data_save_path)

    samples = np.array(samples)
    samples_reshaped = []
    for i in range(samples.shape[1]):
        xs = samples[:, i, :, 0].flatten()
        ys = samples[:, i, :, 1].flatten()
        samples_reshaped.append(np.stack([xs, ys], axis=1))

    samples = np.array(samples_reshaped)*datasets.LEMON_SLICE_STD

    if config.return_dlogp:
        dlogps = np.array(dlogps)
        dlogps_reshaped = []

        for i in range(dlogps.shape[1]):
            dlogps_reshaped.append(dlogps[:, i, :, 0].flatten())

        dlogps = np.array(dlogps_reshaped)

    return samples, dlogps if config.return_dlogp else None


if __name__ == '__main__':
    config = scriptutil.load_config("", "settings_ti.json")  # edit config file to edit run

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    b = torch.load(config.sampling_model, map_location=device, weights_only=False)
    
    loader = scriptutil.get_loaders("test", config, device)
    samples, dlogps = sample(config, b, loader, device=device)

    np.save(os.path.join(config.data_save_path, f'samples_epoch{config.sampling_epoch}_beta_{config.beta_test[1]:.2f}.npy'), samples)
    if config.return_dlogp:
        np.save(os.path.join(config.data_save_path, f'dlogps_epoch{config.sampling_epoch}_beta_{config.beta_test[1]:.2f}.npy'), dlogps)


    if config.return_dlogp:
        # unnormalized importance weights
        ls_settings = {
            "k": 4,
            "gamma": 5,
            "multiplier": 1.2,
        }
        
        sample_weights = weights.statistical_weight(
            x0=samples[0],
            x1=samples[-1],
            dlogp=dlogps[-1],
            beta0=config.beta_test[0],
            beta1=config.beta_test[1],
            ls_settings=ls_settings
        )

        print(f"Effective sample size: {weights.effective_sample_size(sample_weights):.2f}")
        np.save(os.path.join(config.data_save_path, f'weights_epoch{config.sampling_epoch}_beta_{config.beta_test[1]:.2f}.npy'), sample_weights)


    print("\nFinished sampling...\n")
