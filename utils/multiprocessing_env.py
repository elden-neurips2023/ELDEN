# Code is from OpenAI baseline: https://github.com/openai/baselines/tree/master/baselines/common/vec_env

import random
import numpy as np
from multiprocessing import Process, Pipe


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                info["obs"] = ob
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'observation_spec':
            remote.send(env.observation_spec())
        elif cmd == 'obs_delta_range':
            remote.send(env.obs_delta_range())
        elif cmd == 'seed':
            if hasattr(env, "seed"):
                # Standard way to set seed in gym is through this function
                env.seed(data)
            else:
                np.random.seed(data)
                random.seed(data)
        elif cmd.startswith("get_attr_"):
            attr_name = cmd[len("get_attr_"):]
            remote.send(getattr(env, attr_name))
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(object):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True     # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.seed()

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        obs = {key: np.stack([d[key] for d in obs]) for key in obs[0].keys()}
        return obs, np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        obs = {key: np.stack([d[key] for d in obs]) for key in obs[0].keys()}
        return obs

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def observation_spec(self):
        self.remotes[0].send(('observation_spec', None))
        return self.remotes[0].recv()

    def obs_delta_range(self):
        self.remotes[0].send(('obs_delta_range', None))
        return self.remotes[0].recv()

    def __getattr__(self, name):
        self.remotes[0].send(('get_attr_{}'.format(name), None))
        return self.remotes[0].recv()

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True

    def seed(self):
        for i, remote in enumerate(self.remotes):
            remote.send(('seed', i))

    def __len__(self):
        return self.nenvs


def wrap(variable):
    if isinstance(variable, dict):
        return {k: wrap(v) for k, v in variable.items()}
    elif isinstance(variable, (int, float, bool)):
        return np.array([variable])
    elif isinstance(variable, np.ndarray):
        return variable[None]
    else:
        raise ValueError("Unknown variable type:", type(variable))


class SingleVecEnv(object):
    # wrapper to create VecEnv with a single env
    def __init__(self, env, params):
        self.env_name = params.env_params.env_name
        self.env = env
        self.nenvs = 1

    def step(self, action):
        # action: (1, action_dim)
        obs, rew, done, info = self.env.step(action[0])

        if done:
            info["obs"] = obs
            obs = self.env.reset()
        return wrap(obs), wrap(rew), wrap(done), [info]

    def reset(self):
        obs = self.env.reset()
        return wrap(obs)

    def get_save_information(self):
        return self.env.get_save_information()

    def __getattr__(self, name):
        """
        for functions calls to SingleVecEnv, i.e., SingleVecEnv.func(), that are not defined,
        e.g., observation_spec, obs_delta_range, observation_dims, ...
        they will be automatically guided to self.env.func() by this getattr overriding
        """
        return getattr(self.env, name)

    def __len__(self):
        return self.nenvs
