import torch
import numpy as np


def variational_distance(samples, reference, bins=60, low=-2, high=2):
    reference = reference.detach().cpu().numpy()
    samples = samples.detach().cpu().numpy()

    bins = np.linspace(low, high, bins)
    
    ref_hist, _, _ = np.histogram2d(reference[:, 0], reference[:, 1], bins=bins, range=[[low, high], [low, high]])
    ref_hist = ref_hist/ref_hist.sum()

    model_hist, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins, range=[[low, high], [low, high]])
    model_hist = model_hist/model_hist.sum()

    abs_dist = np.abs(ref_hist - model_hist)
    variational_distance = 0.5*abs_dist.flatten().sum()
    return torch.tensor(variational_distance)
