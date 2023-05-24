import torch
import math
import numpy as np


def get_noisy_obs(obs, noise_mag, key=None):
    """
    Takes in obs and return a noisy version of it
    The noise level is copied from the false sample generation
    """
    if key is None:
        obs_keys = ["robot0_eef_pos", "robot0_eef_vel", "robot0_gripper_qpos", "robot0_gripper_qvel",
                     "mov0_pos", "mov0_quat", "unmov0_pos", "unmov0_quat",
                     "marker0_pos"]
    else:
        obs_keys = [key]

    noisy_obs = {}
    for k, v in obs.items():
        if k in obs_keys:
            noisy_obs[k] = v + np.random.uniform(-noise_mag, noise_mag, v.shape)
            # noisy_obs[k] = v + np.random.uniform(0, noise_mag, v.shape)  # We can add noise to one side or both sides
        else:
            noisy_obs[k] = v
    return noisy_obs

def get_random_obs(obs, key_min, key_max, key):
    """
    returns observation with key value randomized across a given range
    """
    obs_keys = [key]

    noisy_obs = {}
    for k, v in obs.items():
        if k in obs_keys:
            noisy_obs[k] = np.random.uniform(key_min, key_max, v.shape)
        else:
            noisy_obs[k] = v
    return noisy_obs


#################################################################
# from https://arxiv.org/pdf/2106.03443.pdf  ####################
def log_2pi():
    return math.log(2 * math.pi)


def gaussian_prod_logconst(m1, v1, m2, v2):
    """Log normalization constant of product of two Gaussians"""
    d = m1.shape[-1]
    v_sum = v1 + v2
    return (-0.5 * (d * log_2pi()
            + torch.sum(torch.log(v_sum), dim=-1)
            + torch.sum((m1 - m2)**2 / v_sum, dim=-1)))


def gaussian_entropy(m, v):
    """Entropy of Gaussian"""
    d = v.shape[-1]
    return 0.5 * (d * (1 + log_2pi()) + torch.sum(torch.log(v), dim=-1))


def kl_div(m1, v1, m2, v2):
    """KL divergence between two Gaussians"""
    d = m1.shape[-1]
    return (0.5 * (-d + ((v1 + (m2 - m1)**2) / v2
            + torch.log(v2) - torch.log(v1)).sum(dim=-1)))


def _kl_div_mixture_app(m1, v1, m2, v2):
    m1_ = m1.unsqueeze(-2)
    v1_ = v1.unsqueeze(-2)

    log_n_mixtures = math.log(m2.shape[-2])

    # Variational approximation
    inner_kls = kl_div(m1_, v1_, m2, v2)
    kls_var = log_n_mixtures - torch.logsumexp(-inner_kls, dim=-1)

    # Product approximation
    log_constants = gaussian_prod_logconst(m1_, v1_, m2, v2)
    kls_prod = (log_n_mixtures - gaussian_entropy(m1, v1)
                - torch.logsumexp(log_constants, dim=-1))
    kls_prod = torch.max(kls_prod, torch.zeros(1))

    kls_app = 0.5 * (kls_var + kls_prod)

    return kls_app, kls_var, kls_prod


def kl_div_mixture_app(m1, v1, m2, v2,
                       return_approximations=False):
    """Approximate KL divergence between Gaussian and mixture of Gaussians
    See Durrieu et al, 2012: "Lower and upper bounds for approximation of the
    Kullback-Leibler divergence between Gaussian Mixture Models"
    https://serval.unil.ch/resource/serval:BIB_513DF4E21898.P001/REF
    Both the variational and the product approximation are simplified here
    compared to the paper, as we assume to have a single Gaussian as the first
    argument.
    m1: ([batch_dims], data_dims)
    v1: ([batch_dims], data_dims)
    m2: ([batch_dims], mixtures, data_dims)
    v2: ([batch_dims], mixtures, data_dims)
    """
    assert m1.ndim + 1 == m2.ndim

    kls_app, kls_var, kls_prod = _kl_div_mixture_app(m1, v1, m2, v2)
    if return_approximations:
        return kls_app, kls_var, kls_prod
    else:
        return kls_app

#################################################################
