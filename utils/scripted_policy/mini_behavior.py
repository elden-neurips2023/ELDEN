import numpy as np


class ScriptedMiniBehavior:
    def __init__(self, env, params):
        self.env = env
        self.random_action_prob = params.scripted_policy_params.random_action_prob

    def reset(self, *args):
        pass

    def act(self, obs):
        if np.random.rand() < self.random_action_prob:
            return np.random.randint(self.env.action_dim)
        else:
            return self.env.generate_action()
