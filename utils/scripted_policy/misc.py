import random
import numpy as np

from utils.utils import DISCRETE_STATE_ENVS
from utils.scripted_policy.craft import ScriptedCraft
from utils.scripted_policy.unlock import ScriptedUnlock
from utils.scripted_policy.mini_behavior import ScriptedMiniBehavior
from utils.scripted_policy.kitchen import ScriptedKitchen
from utils.scripted_policy.kitchen_w_skills import ScriptedKitchenWithSkills
from utils.scripted_policy.pick_place import ScriptedPickAndPlace


def get_scripted_policy(env, params):
    env_name = params.env_params.env_name

    if env_name == "craft":
        script_cls = ScriptedCraft
    elif env_name in DISCRETE_STATE_ENVS:
        script_cls = ScriptedMiniBehavior
    elif env_name == "block":
        script_cls = ScriptedPickAndPlace
    elif env_name == "kitchen":
        script_cls = ScriptedKitchen
    elif env_name == "kitchen_w_skills":
        script_cls = ScriptedKitchenWithSkills
    else:
        raise ValueError("Unknown env_name: {}".format(env_name))

    return ScriptWrapper(env, script_cls, params)


def get_is_demo(step, params, num_envs=0):
    demo_annealing_start = params.scripted_policy_params.demo_annealing_start
    demo_annealing_end = params.scripted_policy_params.demo_annealing_end
    demo_annealing_coef = np.clip((step - demo_annealing_start) / (demo_annealing_end - demo_annealing_start), 0, 1)
    demo_prob_init = params.scripted_policy_params.demo_prob_init
    demo_prob_final = params.scripted_policy_params.demo_prob_final
    demo_prob = demo_prob_init + (demo_prob_final - demo_prob_init) * demo_annealing_coef

    if num_envs:
        return np.random.random(num_envs) < demo_prob
    else:
        return np.random.random() < demo_prob


class ScriptWrapper:
    def __init__(self, env, script_cls, params):
        self.num_envs = params.env_params.num_envs
        self.continuous_action = params.continuous_action
        self.action_dim = params.action_dim
        self.action_spec = params.action_spec

        self.policies = [script_cls(env, params) for _ in range(self.num_envs)]
        self.policies_inited = False

    def reset(self, obs, i=0):
        if not self.policies_inited:
            self.policies_inited = True
            for i in range(self.num_envs):
                obs_i = {key: val[i] for key, val in obs.items()}
                self.policies[i].reset(obs_i)
        else:
            obs_i = {key: val[i] for key, val in obs.items()}
            self.policies[i].reset(obs_i)

    def act(self, obs):
        actions = []
        for i in range(self.num_envs):
            obs_i = {key: val[i] for key, val in obs.items()}
            actions.append(self.policies[i].act(obs_i))
        return np.array(actions)

    def act_randomly(self):
        if self.continuous_action:
            low, high = params.action_spec
            return np.random.uniform(low, high, (self.num_envs, self.action_dim))
        else:
            return np.random.randint(self.action_dim, size=self.num_envs)

    def __getattr__(self, name):
        """
        for functions calls to ScriptWrapper, i.e., ScriptWrapper.func(), that are not defined,
        they will be automatically guided to self.policies[0].func() by this getattr overriding
        """
        assert self.num_envs == 1
        return getattr(self.policies[0], name)
