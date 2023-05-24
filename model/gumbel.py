import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

EPS = 1e-6


def sample_logistic(shape, device):
    u = torch.rand(shape, dtype=torch.float32, device=device)
    u = torch.clip(u, EPS, 1 - EPS)
    return torch.log(u) - torch.log(1 - u)


def gumbel_sigmoid(log_alpha, bs=None, tau=1, hard=True):
    if bs is None:
        shape = log_alpha.shape
    else:
        shape = log_alpha.shape + bs

    logistic_noise = sample_logistic(shape, log_alpha.device)
    y_soft = torch.sigmoid((log_alpha + logistic_noise) / tau)

    if hard:
        y_hard = (y_soft > 0.5).float()
        # This weird line does two things:
        #   1) at forward, we get a hard sample.
        #   2) at backward, we differentiate the gumbel sigmoid
        y = y_hard.detach() - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y


class GumbelMatrix(torch.nn.Module):
    """
    Random matrix M used for the mask. Can sample a matrix and backpropagate using the
    Gumbel straigth-through estimator.
    """
    def __init__(self, shape, init_value, device):
        super(GumbelMatrix, self).__init__()
        self.device = device
        self.shape = shape
        self.log_alpha = torch.nn.Parameter(torch.zeros(shape))
        self.reset_parameters(init_value)

    def forward(self, bs, tau=1, drawhard=True):
        if self.training:
            sample = gumbel_sigmoid(self.log_alpha, self.device, bs, tau=tau, hard=drawhard)
        else:
            sample = (self.log_alpha > 0).float()
        return sample

    def get_prob(self):
        """Returns probability of getting one"""
        return torch.sigmoid(self.log_alpha)

    def reset_parameters(self, init_value):
        log_alpha_init = -np.log(1 / init_value - 1)
        torch.nn.init.constant_(self.log_alpha, log_alpha_init)
