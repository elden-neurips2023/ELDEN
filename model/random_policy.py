import numpy as np
import torch.nn as nn


class RandomPolicy(nn.Module):
    def __init__(self, env):
        super(RandomPolicy, self).__init__()
        self.action_space = env.action_space
        self.num_envs = env.nenvs
        self.num_timesteps = 0

    def act(self, obs):
        return np.array([self.action_space.sample() for _ in range(self.num_envs)])

    def save(self, path):
        pass
