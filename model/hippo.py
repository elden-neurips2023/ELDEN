# Modify from https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/eRL_demo_PPOinSingleFile.py
"""net.py"""

import os
import numpy as np

import torch
import torch.nn as nn

from model.hippo_skills import SkillController
from utils.utils import to_numpy

EPS = 1e-6


class HiPPO(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.policy_params = policy_params = params.policy_params
        self.hippo_params = hippo_params = policy_params.hippo_params

        self.num_envs = num_envs = params.env_params.num_envs
        self.action_dim = params.action_dim
        action_low, action_high = params.action_spec
        self.action_mean, self.action_scale = (action_low + action_high) / 2, (action_high - action_low) / 2

        env_name = params.env_params.env_name
        if "Block" in env_name:
            num_objs = params.env_params.manipulation_env_params.block_env_params.num_movable_objects
        elif "Kitchen" in env_name:
            num_objs = 3
        else:
            raise ValueError("Unknown env_name: {}".format(env_name))

        self.num_objs = hippo_params.num_objs = num_objs

        self.sc = SkillController(params)
        self.skill_set = skill_set = self.sc.skill_set
        self.num_skills = num_skills = len(skill_set[0])
        self.skill_probs = hippo_params.skill_probs


        self.skill_done = np.ones(num_envs, dtype=bool)
        self.prev_skill_idx = [None for _ in range(self.num_envs)]

        global_low, global_high = params.normalization_range
        global_low = np.array(global_low)
        global_high = np.array(global_high)
        self.global_mean = (global_high + global_low) / 2
        self.global_scale = (global_high - global_low) / 2

    def setup_annealing(self, step):
        pass

    def reset(self, i=0):
        self.skill_done[i] = True

    def act(self, obs, deterministic=False):

        obs = {k: v.copy() for k, v in obs.items()}
        for k, v in obs.items():
            if k.endswith("pos") and not k.endswith("qpos"):
                obs[k] = v * self.global_scale + self.global_mean

        for i, done in enumerate(self.skill_done):
            if not done:
                continue
            prev_skill_idx = self.prev_skill_idx[i]
            if prev_skill_idx is not None and self.hippo_params.skill_names[prev_skill_idx] == "pick_place" and self.num_skills > 1:
                # hardcoded to interact with pot after pick_and_place
                skill_idx = self.hippo_params.skill_names.index("lift")
                skill = self.skill_set[i][skill_idx]
                obj = 2
            else:
                if self.skill_probs is None:
                    skill_idx = np.random.randint(self.num_skills)
                else:
                    skill_idx = np.random.choice(self.num_skills, 1, p=self.skill_probs)[0]

                skill = self.skill_set[i][skill_idx]
                object_necessary = skill.is_object_oriented_skill and skill.is_object_necessary
                obj_high = self.num_objs if object_necessary else self.num_objs + 1
                obj = np.random.randint(obj_high)

            skill_param = np.random.rand(skill.num_skill_params) * 2 - 1
            self.prev_skill_idx[i] = skill_idx
            obs_i = {k: v[i] for k, v in obs.items()}
            self.sc.update_config(i, obs_i, skill_idx, obj, skill_param)

        # if self.skill_done:
        #     print("\nskill", self.hippo_params.skill_names[skill_idx])
        #     print("obj", skill.obj_name_mapper.get(skill.obj, "goal"))
        #     print("skill_param", skill_param)
        #     self.skill_step = 0
        #     self.cur_skill = skill
        # else:
        #     self.skill_step += 1
        #     print("{}/{}, {}, {}, {}".format(self.skill_step, self.cur_skill.num_max_steps,
        #                                      getattr(self.cur_skill, "state", None),
        #                                      self.cur_skill.is_disturbed,
        #                                      self.cur_skill.get_disturbance() if hasattr(self.cur_skill, "get_disturbance") else None))

        action, self.skill_done = self.sc.get_action(obs)

        return action

    def act_randomly(self):
        return self.action_mean + self.action_scale * np.random.uniform(-1, 1, (self.num_envs, self.action_dim))

    def save(self, path):
        pass
