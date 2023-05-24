import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import DISCRETE_STATE_ENVS


class IdentityEncoder(nn.Module):
    # extract 1D obs and concatenate them
    def __init__(self, params, key_type="obs_keys"):
        super().__init__()

        self.params = params
        self.keys = [key for key in params[key_type] if params.obs_spec[key].ndim == 1]
        self.feature_dim = np.sum([len(params.obs_spec[key]) for key in self.keys])

        self.index2key = {}
        cum = 0
        for key in self.keys:
            key_dim = len(params.obs_spec[key])
            for i in range(key_dim):
                self.index2key[cum + i] = (key, i)
            cum += key_dim

        self.continuous_state = params.continuous_state
        self.feature_inner_dim = None
        if not self.continuous_state:
            self.feature_inner_dim = np.concatenate([params.obs_dims[key] for key in self.keys])

        self.to(params.device)

    def forward_tensor(self, obs, concat_discrete=False):
        if self.continuous_state:
            obs = torch.cat([obs[k] for k in self.keys], dim=-1)
            return obs
        else:
            obs = [obs_k_i
                   for k in self.keys
                   for obs_k_i in torch.unbind(obs[k], dim=-1)]
            obs = [F.one_hot(obs_i.long(), obs_i_dim).float() if obs_i_dim > 1 else obs_i.unsqueeze(dim=-1)
                   for obs_i, obs_i_dim in zip(obs, self.feature_inner_dim)]
            if concat_discrete:
                obs = torch.cat(obs, dim=-1)
            return obs

    @staticmethod
    def onehot_np(a, num_classes):
        out = np.zeros((a.size, num_classes), dtype=np.float32)
        out[np.arange(a.size), a.ravel()] = 1
        out.shape = a.shape + (num_classes,)
        return out

    def forward_np_array(self, obs, concat_discrete=False):
        if self.continuous_state:
            obs = np.concatenate([obs[k] for k in self.keys], axis=-1)
            return obs
        else:
            obs = [obs_k_i
                   for k in self.keys
                   for obs_k_i in np.moveaxis(obs[k], -1, 0)]
            obs = [self.onehot_np(obs_i.astype(int), obs_i_dim) if obs_i_dim > 1 else obs_i[..., None]
                   for obs_i, obs_i_dim in zip(obs, self.feature_inner_dim)]
            if concat_discrete:
                obs = np.concatenate(obs, axis=-1)
            return obs

    def forward(self, obs, concat_discrete=False):
        if isinstance(obs[self.keys[0]], torch.Tensor):
            return self.forward_tensor(obs, concat_discrete)
        else:
            return self.forward_np_array(obs, concat_discrete)


class ObjectEncoder(nn.Module):
    # extract 1D obs and concatenate them
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.continuous_state = params.continuous_state
        self.env_name = env_name = params.env_params.env_name
        self.continuous_onehot = params.encoder_params.continuous_onehot

        self.keys = [key for key in params.dynamics_keys if params.obs_spec[key].ndim == 1]

        self.feature_dim = np.sum([len(params.obs_spec[key]) for key in self.keys])
        self.feature_inner_dim = None
        if not self.continuous_state:
            self.obs_dims = params.obs_dims
            self.feature_inner_dim = np.concatenate([params.obs_dims[key] for key in self.keys])

        self.obj_keys = []
        self.obj_mapper = {}
        self.obj_dim = {}
        self.obj_inner_dim = {}
        self.obs2obj_index = ()
        # hardcoded obj_mapper to group obs keys to objects
        for key in self.keys:
            if env_name in DISCRETE_STATE_ENVS:
                obj_key = key.split("_")[0]
            else:
                if key.startswith(("robot0_eef", "robot0_gripper")):
                    obj_key = key.split("_")[1]
                else:
                    obj_key = key.split("_")[0]

            if obj_key not in self.obj_mapper:
                self.obj_mapper[obj_key] = []
                self.obj_dim[obj_key] = 0
                self.obj_inner_dim[obj_key] = []
                self.obj_keys.append(obj_key)

            # obs keys that belong to the same object should be next to each other in params.dynamics_keys
            assert self.obj_keys[-1] == obj_key

            self.obj_mapper[obj_key].append(key)
            obj_index = self.obj_keys.index(obj_key)

            if self.continuous_state:
                obs_dim = len(params.obs_spec[key])
                self.obs2obj_index += (obj_index,) * obs_dim
            else:
                obs_dim = len(params.obs_spec[key])
                self.obs2obj_index += (obj_index,) * obs_dim

                obs_inner_dim = params.obs_dims[key]
                obs_dim = sum(obs_inner_dim)
                self.obj_inner_dim[obj_key].extend(obs_inner_dim)
            self.obj_dim[obj_key] += obs_dim

        self.num_objs = len(self.obj_mapper)
        self.to(params.device)

    def forward(self, obs, detach=False):
        if self.continuous_state:
            # overwrite some observations for out-of-distribution evaluation
            if not getattr(self, "manipulation_train", True):
                test_scale = self.manipulation_test_scale
                obs = {k: torch.randn_like(v) * test_scale if "marker" in k else v
                       for k, v in obs.items()}
            obs = [torch.cat([obs[obs_k] for obs_k in self.obj_mapper[obj_k]], dim=-1)
                   for obj_k in self.obj_keys]
            return obs
        else:
            feature = []
            for obj_k in self.obj_keys:
                obj_obs = []
                for obs_k in self.obj_mapper[obj_k]:
                    obs_v = torch.unbind(obs[obs_k], dim=-1)
                    obs_v_dims = self.obs_dims[obs_k]
                    for obs_i, obs_i_dim in zip(obs_v, obs_v_dims):
                        obs_v = F.one_hot(obs_i.long(), obs_i_dim).float()
                        obj_obs.append(obs_v)
                obj_obs = torch.cat(obj_obs, dim=-1)
                if self.continuous_onehot:
                    obj_obs = obj_obs + torch.rand_like(obj_obs)
                feature.append(obj_obs)
            return feature
