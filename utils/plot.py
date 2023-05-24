import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils.utils import to_numpy


def plot_adjacency_intervention_mask(params, model, writer, step):
    mask = model.get_mask(bool_mask=False)
    if mask is None:
        return
    thre = model.get_threshold()
    mask = to_numpy(mask)

    obs_keys = params.dynamics_keys
    obs_spec = params.obs_spec
    feature_dim = params.feature_dim
    action_dim = params.action_dim
    num_objs = params.num_objs
    obj_keys = params.obj_keys

    assert mask.ndim in [2, 3]
    if mask.ndim == 2:
        mask = mask[None]
    num_masks, num_rows, num_cols = mask.shape

    if not params.continuous_action:
        action_dim = 1
    assert num_rows in [feature_dim, num_objs]
    assert num_cols in [feature_dim + action_dim, num_objs + 1]

    vmax = thre
    while vmax < 1:
        vmax = vmax * 10
        mask = mask * 10

    for i in range(num_masks):
        fig = plt.figure(figsize=(num_cols * 0.45 + 2, num_rows * 0.45 + 2))
        sns.heatmap(mask[i], linewidths=1, vmin=0, vmax=vmax, square=True, annot=True, fmt='.2f', cbar=False)

        ax = plt.gca()
        ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)

        tick_loc = []
        cum_idx = 0
        for k in obs_keys:
            obs_dim = obs_spec[k].shape[0]
            tick_loc.append(cum_idx + obs_dim * 0.5)
            cum_idx += obs_dim
            if num_cols == feature_dim + action_dim:
                plt.vlines(cum_idx, ymin=0, ymax=num_rows, colors='blue', linewidths=2)
            if num_rows == feature_dim and k != obs_keys[-1]:
                plt.hlines(cum_idx, xmin=0, xmax=num_cols, colors='blue', linewidths=2)

        if num_rows == feature_dim:
            plt.yticks(tick_loc, obs_keys, rotation=0)
        else:
            plt.yticks(np.arange(num_objs) + 0.5, obj_keys, rotation=0)

        if num_cols == feature_dim + action_dim:
            plt.xticks(tick_loc + [feature_dim + 0.5 * action_dim], obs_keys + ["action"], rotation=90)
        else:
            plt.xticks(np.arange(num_objs + 1) + 0.5, obj_keys + ["action"], rotation=90)

        fig.tight_layout()
        writer.add_figure("adjacency {}".format(i), fig, step + 1)
    plt.close("all")
