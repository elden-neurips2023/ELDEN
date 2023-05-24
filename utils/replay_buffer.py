import os
import torch
import numpy as np

from utils.utils import preprocess_obs, to_numpy
from utils.sum_tree import BatchSumTree


def take(array, start, end):
    if start >= end:
        end += len(array)
    idxes = np.arange(start, end) % len(array)
    return array[idxes]


def assign(array, start, end, value):
    if start >= end:
        end += len(array)
    idxes = np.arange(start, end) % len(array)
    array[idxes] = value


class ReplayBuffer:
    """Buffer to store environment transitions."""

    def __init__(self, params):
        self.params = params
        self.device = params.device
        self.continuous_action = params.continuous_action

        training_params = params.training_params
        replay_buffer_params = training_params.replay_buffer_params

        self.capacity = capacity = replay_buffer_params.capacity
        self.saving_dir = params.replay_buffer_dir

        dynamics_params = params.dynamics_params
        self.dynamics_batch_size = dynamics_params.batch_size
        self.num_dynamics_pred_steps = dynamics_params.num_pred_steps
        self.ensemble_size = dynamics_params.ensemble_size
        self.monolithic_arch = dynamics_params.monolithic_arch
        self.ensemble_use_diff_batches = dynamics_params.ensemble_use_diff_batches

        self.policy_batch_size = params.policy_params.batch_size
        self.num_policy_td_steps = params.policy_params.num_td_steps

        # init data
        obs_spec = params.obs_spec
        action_dim = params.action_dim
        self.obses = {k: np.empty((capacity, *v.shape), dtype=v.dtype) for k, v in obs_spec.items()}
        if self.continuous_action:
            self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        else:
            self.actions = np.empty((capacity, 1), dtype=np.int64)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
        self.is_trains = np.empty((capacity, 1), dtype=bool)

        self.store_gt_mask = training_params.use_gt_causality and params.mask_gt_params.intrinsic_reward_type == "gt_causal_curiosity"
        if self.store_gt_mask:
            self.gt_mask = np.empty((capacity, params.feature_dim, params.feature_dim + 1), dtype=np.float32)

        # init writing
        self.idx = 0
        self.last_save = 0
        self.full = False

        # loading
        self.load(training_params.load_replay_buffer)

        # cache for vecenv
        self.num_envs = params.env_params.num_envs
        self.temp_buffer = [[] for _ in range(self.num_envs)]

        self.init_cache()

    def init_cache(self):
        # cache for faster query
        if self.ensemble_use_diff_batches:
            dynamics_train_tiling = (self.dynamics_batch_size, self.ensemble_size, 1)
        else:
            dynamics_train_tiling = (self.dynamics_batch_size, 1)
        self.dynamics_train_idxes_base = np.tile(np.arange(self.num_dynamics_pred_steps), dynamics_train_tiling)
        self.dynamics_eval_idxes_base = np.tile(np.arange(self.num_dynamics_pred_steps), (self.dynamics_batch_size, 1))

        if self.num_policy_td_steps == 1:
            self.policy_idxes_base = 0
        else:
            self.policy_idxes_base = np.tile(np.arange(self.num_policy_td_steps), (self.policy_batch_size, 1))

    def add(self, obs, action, reward, next_obs, done, is_train, info):
        for i in range(self.num_envs):
            if self.store_gt_mask:
                obs_i = {key: val[i] for key, val in obs.items()}
                self.temp_buffer[i].append([obs_i, action[i], reward[i], done[i], is_train[i], info[i]["local_causality"]])
                if done[i]:
                    for obs_, action_, reward_, done_, is_train_, gt_mask_ in self.temp_buffer[i]:
                        self._add(obs_, action_, reward_, done_, is_train_, gt_mask_)
                    final_obs = info[i]["obs"]
                    # use done = -1 as a special indicator that the added obs is the last obs in the episode
                    self._add(final_obs, action_, 0, -1, is_train_, gt_mask_)
                    self.temp_buffer[i] = []
            else:
                obs_i = {key: val[i] for key, val in obs.items()}
                self.temp_buffer[i].append([obs_i, action[i], reward[i], done[i], is_train[i]])
                if done[i]:
                    for obs_, action_, reward_, done_, is_train_ in self.temp_buffer[i]:
                        self._add(obs_, action_, reward_, done_, is_train_)
                    final_obs = info[i]["obs"]
                    # use done = -1 as a special indicator that the added obs is the last obs in the episode
                    self._add(final_obs, action_, 0, -1, is_train_)
                    self.temp_buffer[i] = []

    def _add(self, obs, action, reward, done, is_train, gt_mask=None):
        obs = preprocess_obs(obs, self.params)
        for k in obs.keys():
            np.copyto(self.obses[k][self.idx], obs[k])

        if self.continuous_action and action.dtype != np.float32:
            action = action.astype(np.float32)
        elif not self.continuous_action:
            action = np.int64(action)

        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.dones[self.idx], done)
        np.copyto(self.is_trains[self.idx], is_train)

        if self.store_gt_mask:
            assert gt_mask is not None
            np.copyto(self.gt_mask[self.idx], gt_mask)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def valid_idx(self, idx, num_steps, type, use_part="all"):
        if use_part != "all":
            is_train = self.is_trains[idx]
            if use_part == "train" and not is_train:
                return False
            if use_part == "eval" and is_train:
                return False

        cross_episode = (self.dones[idx:idx + num_steps] == -1).any()

        # self.idx - 1 is the latest data point
        # idx is the first data point to use, idx + num_steps is the last data point to use (both inclusive)
        cross_newest_data = (idx < self.idx) and (idx + num_steps >= self.idx)

        return not (cross_episode or cross_newest_data)

    def sample_idx(self, type, use_part="all"):
        if type == "dynamics" or type == "causal_gt":
            num_steps = self.num_dynamics_pred_steps
            if self.ensemble_use_diff_batches and use_part == "train":
                batch_size = (self.dynamics_batch_size, self.ensemble_size)
            else:
                batch_size = (self.dynamics_batch_size,)
        elif type == "policy":
            num_steps = self.num_policy_td_steps
            batch_size = [self.policy_batch_size]
        else:
            raise NotImplementedError

        idxes = []
        for _ in range(np.prod(batch_size)):
            while True:
                idx = np.random.randint(len(self) - num_steps)

                if self.valid_idx(idx, num_steps, type, use_part):
                    idxes.append(idx)
                    break

        return np.array(idxes).reshape(batch_size)

    def get_idxes_base(self, type, use_part):
        if type == "dynamics" or type == "causal_gt":
            if use_part == "train":
                return self.dynamics_train_idxes_base
            else:
                return self.dynamics_eval_idxes_base
        elif type == "policy":
            return self.policy_idxes_base
        else:
            raise NotImplementedError

    def construct_transition(self, idxes, type, use_part):
        pred_idxes_base = self.get_idxes_base(type, use_part)

        if type == "policy":
            obs_idxes = idxes                                           # (batch_size,)
            if self.num_policy_td_steps > 1:
                idxes = idxes[..., None]                                # (batch_size, 1)
        elif type == "dynamics" or type == "causal_gt":
            obs_idxes = idxes                                           # (batch_size,) or (batch_size, ensemble_size)
            idxes = idxes[..., None]
        else:
            raise NotImplementedError

        next_obs_idxes = pred_idxes_base + idxes + 1                    # (batch_size,) or (batch_size, num_pred_steps)
        act_rew_done_idxes = next_obs_idxes - 1                         # (batch_size,) or (batch_size, num_pred_steps)

        obses = {k: torch.tensor(v[obs_idxes], device=self.device) for k, v in self.obses.items()}

        actions = torch.tensor(self.actions[act_rew_done_idxes],
                               dtype=torch.float32 if self.continuous_action else torch.int64, device=self.device)

        next_obses = {k: torch.tensor(v[next_obs_idxes], device=self.device) for k, v in self.obses.items()}

        rewards = dones = None
        if type == "policy":
            rewards = torch.tensor(self.rewards[act_rew_done_idxes], dtype=torch.float32, device=self.device)

        if type == "policy":
            dones = torch.tensor(self.dones[act_rew_done_idxes], dtype=torch.float32, device=self.device)

        if type == "causal_gt" and self.store_gt_mask:
            # This is a bit hacky: basically we replace the next obs with causal gt
            next_obses = torch.tensor(self.gt_mask[act_rew_done_idxes], dtype=torch.float32, device=self.device)

        return obses, actions, rewards, next_obses, dones

    def sample(self, type, use_part="all"):
        """
        Sample training data for dynamics model
        return:
            obses: (batch_size, obs_spec)
            actions: (batch_size, action_dim) or (batch_size, num_steps, action_dim)
            next_obses: (batch_size, action_dim) or (batch_size, num_steps, obs_spec)
        """
        idxes = self.sample_idx(type, use_part)
        obses, actions, rewards, next_obses, dones = self.construct_transition(idxes, type, use_part)
        return obses, actions, rewards, next_obses, dones, idxes

    def sample_dynamics(self, use_part="all"):
        obses, actions, _, next_obses, _, idxes = self.sample("dynamics", use_part)
        return obses, actions, next_obses, idxes

    def sample_causal_gt(self, use_part="all"):
        obses, actions, _, causal_graph, _, idxes = self.sample("causal_gt", use_part)
        return obses, actions, causal_graph, idxes

    def sample_policy(self):
        obses, actions, rewards, next_obses, dones, _ = self.sample("policy", use_part="all")
        return obses, actions, rewards, next_obses, dones

    def save(self):
        assert self.idx != self.last_save

        for chunk in os.listdir(self.saving_dir):
            start, end = [int(x) for x in chunk.split(".")[0].split("_")]
            if (self.last_save < end or (end < start < self.last_save)) and (self.idx >= end or (self.idx < self.last_save < end)):
                chunk = os.path.join(self.saving_dir, chunk)
                os.remove(chunk)

        chunk = "%d_%d.p" % (self.last_save, self.idx)
        path = os.path.join(self.saving_dir, chunk)

        payload = {"obses": {k: take(v, self.last_save, self.idx) for k, v in self.obses.items()},
                   "actions": take(self.actions, self.last_save, self.idx),
                   "rewards": take(self.rewards, self.last_save, self.idx),
                   "dones": take(self.dones, self.last_save, self.idx),
                   "is_trains": take(self.is_trains, self.last_save, self.idx)}

        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        if save_dir is None or not os.path.isdir(save_dir):
            return

        chunks = [os.path.join(save_dir, chunk) for chunk in os.listdir(save_dir)]
        chunks.sort(key=os.path.getctime)
        for chunk in chunks:
            chunk_fname = os.path.split(chunk)[1]
            start, end = [int(x) for x in chunk_fname.split(".")[0].split("_")]
            payload = torch.load(chunk)
            for k in self.obses:
                assign(self.obses[k], start, end, payload["obses"][k])
            assign(self.actions, start, end, payload["actions"])
            assign(self.rewards, start, end, payload["rewards"])
            assign(self.dones, start, end, payload["dones"])
            assign(self.is_trains, start, end, payload["is_trains"])

            self.idx = end
            if end < start or end == self.capacity:
                self.full = True

            print("loaded", chunk)

        if len(chunks):
            # episode ends
            self.dones[self.idx - 1] = -1

        print("replay buffer loaded from", save_dir)

    def __len__(self):
        return self.capacity if self.full else (self.idx + 1)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, params):
        replay_buffer_params = params.training_params.replay_buffer_params
        capacity = replay_buffer_params.capacity

        dynamics_params = params.dynamics_params
        self.dynamics_batch_size = dynamics_params.batch_size
        self.ensemble_size = dynamics_params.ensemble_size
        self.monolithic_arch = dynamics_params.monolithic_arch
        self.ensemble_use_diff_batches = dynamics_params.ensemble_use_diff_batches

        self.feature_dim = params.feature_dim
        self.num_trees = 1
        self.dynamics_batch_shape = [self.dynamics_batch_size]
        if self.ensemble_use_diff_batches:
            self.num_trees *= self.ensemble_size
            self.dynamics_batch_shape.append(self.ensemble_size)
        if not self.monolithic_arch:
            self.num_trees *= self.feature_dim
            self.dynamics_batch_shape.append(self.feature_dim)

        self.dynamics_train_tree = BatchSumTree(self.num_trees, capacity, np.prod(self.dynamics_batch_size))

        self.dynamics_alpha = replay_buffer_params.dynamics_alpha
        self.max_priority = 1

        self.dynamics_to_add_buffer = []

        super(PrioritizedReplayBuffer, self).__init__(params)

    def init_cache(self):
        # cache for faster query
        super().init_cache()
        self.dynamics_train_idxes_base = np.tile(np.arange(self.num_dynamics_pred_steps), self.dynamics_batch_shape + [1])

    def _add(self, obs, action, reward, done, is_train):
        super(PrioritizedReplayBuffer, self)._add(obs, action, reward, done, is_train)
        self._add_to_tree(done, is_train)

    def _add_to_tree(self, done, is_train):
        dynamics_prob = self.max_priority * (is_train and done != -1)

        if self.params.training_params.num_dynamics_opt_steps:
            self.dynamics_to_add_buffer.append(dynamics_prob)
            if done == -1:
                if self.num_dynamics_pred_steps > 1:
                    raise NotImplementedError("need to change sampling probabilities of previous transitions")
                self.dynamics_train_tree.add(np.array(self.dynamics_to_add_buffer))
                self.dynamics_to_add_buffer = []

    def update_priorities(self, idxes, probs, type):
        if isinstance(probs, torch.Tensor):
            probs = to_numpy(probs)

        min_prob = 0.05

        if type == "dynamics":
            assert idxes.size == probs.size == np.prod(self.dynamics_batch_shape)
            if idxes.shape != probs.shape:
                probs = probs.reshape(self.dynamics_batch_size, -1).T
            probs = np.clip(probs ** self.dynamics_alpha, min_prob, self.max_priority)
            tree = self.dynamics_train_tree
        else:
            raise NotImplementedError

        tree.update(idxes, probs)

    def sample(self, type, use_part="all"):
        if type == "dynamics":
            num_steps = self.num_dynamics_pred_steps
            batch_size = self.dynamics_batch_size
        elif type == "policy":
            num_steps = self.num_policy_td_steps
            batch_size = self.policy_batch_size
        else:
            raise NotImplementedError

        tree_idxes, data_idxes = self.sample_idx(batch_size, num_steps, type, use_part)
        obses, actions, rewards, next_obses, dones = self.construct_transition(data_idxes, type, use_part)

        return obses, actions, rewards, next_obses, dones, tree_idxes

    def sample_idx(self, batch_size, num_steps, type, use_part="all"):
        if type == "policy" or use_part == "eval":
            idxes = super().sample_idx(type, use_part)
            return None, idxes

        assert use_part != "all"
        if type == "dynamics":
            tree = self.dynamics_train_tree
        else:
            raise NotImplementedError

        return self.sample_idx_from_tree(tree, batch_size, num_steps)

    def sample_idx_from_tree(self, tree, batch_size, num_steps):
        segment = tree.total() / batch_size       # scalar or (feature_dim,)
        if not self.full:
            # - self.max_priority * num_steps to avoid infinite loop of sampling the newly added sample
            segment -= self.max_priority * num_steps / batch_size

        if isinstance(tree, BatchSumTree):
            s = np.random.uniform(size=(self.num_trees, batch_size)) + np.arange(batch_size)
            s = s * segment[:, None]                                    # (self.num_trees, batch_size)
        else:
            raise NotImplementedError

        tree_idxes, data_idxes = tree.get(s)                            # (self.num_trees, batch_size)
        ooi_mask = data_idxes + num_steps >= len(self)
        if ooi_mask.any():
            data_idxes[ooi_mask] = np.random.randint(len(self) - num_steps)
            tree_idxes[ooi_mask] = data_idxes[ooi_mask] + self.capacity - 1
        data_idxes = data_idxes.T.reshape(self.dynamics_batch_shape)

        return tree_idxes, data_idxes

    def save(self):
        super().save()
        path = os.path.join(self.saving_dir, "priorities.p")

        payload = {"dynamics": self.dynamics_train_tree.trees[:, self.capacity - 1:]}

        torch.save(payload, path)

    def load(self, save_dir):
        super().load(save_dir)

        if save_dir is not None and os.path.exists(os.path.join(save_dir, "priorities.p")):
            payload = torch.load(os.path.join(save_dir, "priorities.p"))
            dynamics_priorities = payload["dynamics"]
        else:
            num_data = self.capacity if self.full else self.idx
            dones = self.dones[:num_data, 0]
            is_trains = self.is_trains[:num_data, 0]

            valid_mask = np.array([(dones[i:i + self.num_dynamics_pred_steps] != -1).all()
                                   for i in range(num_data)])
            dynamics_priorities = valid_mask * is_trains * self.max_priority

        self.dynamics_train_tree.init_trees(dynamics_priorities)
