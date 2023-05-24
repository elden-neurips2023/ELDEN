import os
import copy
import time
import json
import torch
import shutil
import random
import numpy as np

import gym
import mini_behavior

import os.path as osp

# from env.physical_env import Physical
from env.minecraft2d import CraftWorld
from env.block import Block
from env.kitchen import Kitchen
from env.kitchen_w_skills import KitchenWithSkill
from utils.multiprocessing_env import SubprocVecEnv, SingleVecEnv

from robosuite.controllers import load_controller_config

DISCRETE_STATE_ENVS = ["craft", "installing_printer", "thawing", "clearning_car", "empty"]
DISCRETE_ACTION_ENVS = ["craft", "installing_printer", "thawing", "clearning_car", "kitchen_w_skills", "empty"]
TASK_LEARNING_ALGO_NAMES = ["ppo", "dqn", "sac"]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class TrainingParams(AttrDict):
    def __init__(self, training_params_fname="params.json", train=True):
        self.load(training_params_fname)

        save_load_path = getattr(self.training_params, "save_load_path", None)
        local_save_load_path = osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))
        if save_load_path is None or not os.access(save_load_path, os.W_OK):
            save_load_path = local_save_load_path

        redirect_params_fname = osp.join(save_load_path, "causal_gradient_reg", "policy_params.json")
        if train and osp.exists(redirect_params_fname) and not osp.samefile(training_params_fname, redirect_params_fname):
            print("Redict to loading parameters from", redirect_params_fname)
            training_params_fname = redirect_params_fname
            self.load(training_params_fname)

        if getattr(self.training_params, "redirect_save", False):
            local_save_load_path = save_load_path

        training_params = self.training_params
        if training_params.load_dynamics is not None:
            training_params.load_dynamics = \
                osp.join(save_load_path, "interesting_models", training_params.load_dynamics)
        if training_params.load_policy is not None:
            training_params.load_policy = \
                osp.join(save_load_path, "interesting_models", training_params.load_policy)
        if training_params.load_replay_buffer is not None:
            training_params.load_replay_buffer = \
                osp.join(save_load_path, "replay_buffer", training_params.load_replay_buffer)

        if train:
            info = self.info.replace(" ", "_")
            experiment_dirname = info + "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
            self.rslts_dir = osp.join(local_save_load_path, "rslts", self.sub_dirname, experiment_dirname)
            os.makedirs(self.rslts_dir)
            shutil.copyfile(training_params_fname, osp.join(self.rslts_dir, "params.json"))

            self.replay_buffer_dir = None
            if training_params.collect_transitions:
                self.replay_buffer_dir = osp.join(local_save_load_path, "replay_buffer", experiment_dirname)
                os.makedirs(self.replay_buffer_dir)

        super(TrainingParams, self).__init__(self.__dict__)

    def load(self, params_fname):
        config = json.load(open(params_fname))
        for k, v in config.items():
            self.__dict__[k] = v
        self.__dict__ = self._clean_dict(self.__dict__)

    def _clean_dict(self, _dict):
        for k, v in _dict.items():
            if v == "":  # encode empty string as None
                v = None
            if isinstance(v, dict):
                v = self._clean_dict(v)
            _dict[k] = v
        return AttrDict(_dict)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_numpy(tensor_data):
    if isinstance(tensor_data, dict):
        return {k: to_numpy(v) for k, v in tensor_data.items()}
    elif isinstance(tensor_data, torch.Tensor):
        return tensor_data.detach().cpu().numpy()
    else:
        return tensor_data


def to_float(x):
    if isinstance(x, (float, int)):
        return x
    else:
        return x.item()


def to_device(dictionary, device):
    """
    place dict of tensors + dict to device recursively
    """
    new_dictionary = {}
    for key, val in dictionary.items():
        if isinstance(val, dict):
            new_dictionary[key] = to_device(val, device)
        elif isinstance(val, torch.Tensor):
            new_dictionary[key] = val.to(device)
        else:
            raise ValueError("Unknown value type {} for key {}".format(type(val), key))
    return new_dictionary


def preprocess_obs(obs, params, key_type="obs_keys"):
    """
    filter unused obs keys, convert to np.float32 / np.uint8, resize images if applicable
    """
    def to_type(ndarray, type):
        if ndarray.dtype != type:
            ndarray = ndarray.astype(type)
        return ndarray

    obs_spec = getattr(params, "obs_spec", obs)
    new_obs = {}

    keys = params[key_type]

    for k in keys:
        val = obs[k]
        val_spec = obs_spec[k]
        if isinstance(val_spec, np.ndarray):
            if val_spec.ndim == 1:
                val = to_type(val, np.float32)
            elif val_spec.ndim == 3:
                num_channel = val.shape[2]
                if num_channel == 1:
                    val = to_type(val, np.float32)
                elif num_channel == 3:
                    val = to_type(val, np.uint8)
                else:
                    raise NotImplementedError
                val = val.transpose((2, 0, 1))                  # c, h, w
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        new_obs[k] = val
    return new_obs


def update_obs_act_spec(env, params):
    """
    get act_dim and obs_spec from env and add to params
    """
    env_params = params.env_params
    env_name = params.env_params.env_name

    params.continuous_state = env_name not in DISCRETE_STATE_ENVS
    params.continuous_action = env_name not in DISCRETE_ACTION_ENVS

    params.action_spec = env.action_spec if params.continuous_action else None
    params.action_dim = env.action_dim

    # get obs key
    if params.continuous_state:
        if env_name == "kitchen_w_skills":
            env_name = "kitchen"
            params.action_dim = env.num_skills
        env_specific_params = env_params.manipulation_env_params[env_name + "_env_params"]
    else:
        env_specific_params = env_params[env_name + "_env_params"]

    env.horizon = env_specific_params.horizon

    if env_name == "craft":
        params.dynamics_keys = env.dynamics_keys
    else:
        params.dynamics_keys = env_specific_params.dynamics_keys

    params.obs_keys = params.policy_keys = env_params.policy_keys + params.dynamics_keys + env_specific_params.policy_additional_keys
    params.obs_spec = preprocess_obs(env.observation_spec(), params)

    if params.continuous_state:
        params.obs_dims = None
        obs_delta_range = env.obs_delta_range()
        obs_delta_low, obs_delta_high = {k: v[0] for k, v in obs_delta_range.items()}, {k: v[1] for k, v in obs_delta_range.items()}
        obs_delta_low = preprocess_obs(obs_delta_low, params, key_type="dynamics_keys")
        obs_delta_high = preprocess_obs(obs_delta_high, params, key_type="dynamics_keys")
        params.obs_delta_range = {k: [torch.from_numpy(obs_delta_low[k]).to(params.device),
                                      torch.from_numpy(obs_delta_high[k]).to(params.device)]
                                  for k in obs_delta_low}
        params.normalization_range = env_specific_params.normalization_range
    else:
        params.obs_dims = env.observation_dims()


def get_single_env(params, render=False):
    env_params = params.env_params
    env_name = env_params.env_name
    if env_name == "craft":
        env = CraftWorld(params)
    elif env_name == "empty":
        env_specific_params = env_params[env_name + "_env_params"]
        env = gym.make("MiniGrid-Empty-v0", size=env_specific_params["size"], max_steps=env_specific_params["horizon"])
    elif env_name in DISCRETE_STATE_ENVS:
        env_specific_params = env_params[env_name + "_env_params"]
        env_id = "MiniGrid-" + env_name + "-v0"
        kwargs = {"room_size": env_specific_params.room_size,
                  "max_steps": env_specific_params.horizon,
                  "use_stage_reward": env_params.use_stage_reward}
        if env_name == "clearning_car":
            kwargs["add_noisy_tv"] = env_specific_params.noisy_tv
            kwargs["tv_dim"] = env_specific_params.tv_dim
            kwargs["tv_channel"] = env_specific_params.tv_channel
        env = gym.make(env_id, **kwargs)
    else:
        env_dict = {"block": Block,
                    "kitchen": Kitchen,
                    "kitchen_w_skills": KitchenWithSkill}
        Env = env_dict[env_name]

        if env_name == "kitchen_w_skills":
            env_name = "kitchen"

        manipulation_env_params = env_params.manipulation_env_params
        env_kwargs = copy.deepcopy(manipulation_env_params[env_name + "_env_params"])
        env_kwargs.pop("dynamics_keys")
        env_kwargs.pop("policy_additional_keys")

        env = Env(robots=manipulation_env_params.robots,
                  controller_configs=load_controller_config(default_controller=manipulation_env_params.controller_name),
                  gripper_types=manipulation_env_params.gripper_types,
                  has_renderer=render,
                  has_offscreen_renderer=False,
                  use_camera_obs=False,
                  ignore_done=False,
                  control_freq=manipulation_env_params.control_freq,
                  reward_scale=manipulation_env_params.reward_scale,
                  **env_kwargs)
    return env


def get_subproc_env(params):
    def get_single_env_wrapper():
        return get_single_env(params)
    return get_single_env_wrapper


def get_env(params, render=False):
    num_envs = params.env_params.num_envs
    if render:
        assert num_envs == 1
    if num_envs == 1:
        return SingleVecEnv(get_single_env(params, render), params)
    else:
        return SubprocVecEnv([get_subproc_env(params) for _ in range(num_envs)])


def get_start_step_from_model_loading(params):
    """
    if model-based policy is loaded, return its training step;
    elif dynamics is loaded, return its training step;
    else, return 0
    """
    task_learning = params.training_params.policy_algo in TASK_LEARNING_ALGO_NAMES
    load_dynamics = params.training_params.load_dynamics
    load_policy = params.training_params.load_policy
    if load_policy is not None and osp.exists(load_policy):
        model_name = load_policy.split(os.sep)[-1]
        start_step = int(model_name.split("_")[-1])
        print("start_step:", start_step)
    elif load_dynamics is not None and osp.exists(load_dynamics) and not task_learning:
        model_name = load_dynamics.split(os.sep)[-1]
        start_step = int(model_name.split("_")[-1])
        print("start_step:", start_step)
    else:
        start_step = 0
    return start_step


def obs_dict_to_features(obs, params, device, policy_encoder, tensorize):
    with torch.no_grad():
        processed_obs = preprocess_obs(obs, params, key_type="policy_keys")
        if tensorize:
            processed_obs = {k: torch.tensor(v, device=device) for k, v in processed_obs.items()}
        obs_feature = policy_encoder(processed_obs, concat_discrete=True)
    return obs_feature


def safe_mean(arr) -> np.ndarray:
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr: Numpy array or list of values
    :return:
    """
    return np.nan if len(arr) == 0 else np.mean(arr)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
