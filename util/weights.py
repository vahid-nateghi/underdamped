import numpy as np
import Lemon_Slice as ls


def model_prob(x0, dlogp, beta0, ls_settings=None):
    """## Unnormalized probability of a configuration x0 under aTI model.

    ### Args:
        - `x0 (_type_)`: initial configuration
        - `dlogp (_type_)`: log probability difference from the model
        - `beta0 (_type_)`: inverse temperature of the initial state
        - `ls_settings (_type_, optional)`: settings for the Lemon Slice potential. Defaults to None.
    """
    if ls_settings is None:
        ls0 = ls.LemonSlice(beta=beta0)
    else:
        ls0 = ls.LemonSlice(k=ls_settings["k"], beta=beta0, gamma=ls_settings["gamma"], multiplier=ls_settings["multiplier"])

    u0 = ls0.potential(x0.T)
    return np.exp(-beta0*u0 - dlogp)


def boltzmann_prob(x1, beta1, ls_settings=None):
    """## Unnormalized Boltzmann probability of a configuration x1.
    ### Args:
        - `x1 (_type_)`: final configuration
        - `beta1 (_type_)`: inverse temperature of the final state
        - `ls_settings (_type_, optional)`: settings for the Lemon Slice potential. Defaults to None.
    """
    if ls_settings is None:
        ls1 = ls.LemonSlice(beta=beta1)
    
    else:
        ls1 = ls.LemonSlice(k=ls_settings["k"], beta=beta1, gamma=ls_settings["gamma"], multiplier=ls_settings["multiplier"])
    
    u1 = ls1.potential(x1.T)
    return np.exp(-beta1*u1)


def statistical_weight(x0, x1, dlogp, beta0, beta1, ls_settings=None):
    p1_model = model_prob(x0, dlogp, beta0, ls_settings)
    p1_target = boltzmann_prob(x1, beta1, ls_settings)
    return p1_target/p1_model


def effective_sample_size(weights):
    """## Effective sample size of a set of weights.
    ### Args:
        - `weights (_type_)`: weights to compute the effective sample size of
    """
    return ((np.sum(weights))**2/np.sum(weights**2))/len(weights)
