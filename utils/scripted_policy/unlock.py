import numpy as np


class ScriptedUnlock:
    def __init__(self, env, params):
        self.env = env

    def reset(self, *args):
        pass

    def act(self, obs):
        return self.env.generate_action()
