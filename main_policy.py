import os
import time
import numpy as np

from gym.spaces import Box, Discrete

import torch
from torch.utils.tensorboard import SummaryWriter

from model.encoder import IdentityEncoder, ObjectEncoder
from model.ensemble_dynamics_grad_attn import EnsembleDynamicsGradAttn
from model.supervised_causal_predictor import EnsembleCausalPredictor

from model.hippo import HiPPO
from model.random_policy import RandomPolicy

from utils.utils import TrainingParams, TASK_LEARNING_ALGO_NAMES
from utils.utils import get_env, update_obs_act_spec
from utils.utils import preprocess_obs, set_seed_everywhere, get_start_step_from_model_loading, to_float, to_numpy
from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from utils.plot import plot_adjacency_intervention_mask
from utils.scripted_policy.misc import get_scripted_policy, get_is_demo
from stable_baselines3 import PPO, SAC, DQN
from utils.utils import obs_dict_to_features, safe_mean
from collections import deque


np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)
torch.set_default_tensor_type(torch.FloatTensor)


def train(params):
    device = torch.device("cuda:{}".format(params.cuda_id) if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")

    set_seed_everywhere(params.seed)

    params.device = device
    env_params = params.env_params
    training_params = params.training_params
    replay_buffer_params = training_params.replay_buffer_params

    # init environment
    render = False
    num_envs = params.env_params.num_envs
    env = get_env(params, render)

    # init model
    update_obs_act_spec(env, params)
    dynamics_encoder = IdentityEncoder(params, key_type="dynamics_keys")
    policy_encoder = IdentityEncoder(params, key_type="policy_keys")
    obj_encoder = ObjectEncoder(params)
    params.feature_dim = dynamics_encoder.feature_dim
    params.num_objs = obj_encoder.num_objs
    params.obj_keys = obj_encoder.obj_keys

    # gt causality should replace dynamics: we don't need dynamics in this case
    if training_params.use_gt_causality:
        assert params.dynamics_params.ensemble_size == 1  # This is to make sure replay buffer functions properly
        pred_algo = training_params.gt_causality_algo
        if pred_algo == "grad_attn":
            Dynamics = EnsembleCausalPredictor
        else:
            raise NotImplementedError
    else:
        # dynamics (should automatically load)
        dynamics_algo = training_params.dynamics_algo
        if dynamics_algo == "grad_attn":
            Dynamics = EnsembleDynamicsGradAttn
        else:
            raise NotImplementedError
    dynamics = Dynamics(dynamics_encoder, obj_encoder, params)

    num_dynamics_opt_steps = training_params.num_dynamics_opt_steps
    if dynamics.intrinsic_reward_type != "none" and not training_params.use_gt_causality:
        assert dynamics_algo == "grad_attn" and num_dynamics_opt_steps

    # policy
    # Add API necessary for sb3
    env.num_envs = num_envs
    if params.continuous_state:
        env.observation_space = Box(low=-1.0, high=1.0, shape=(policy_encoder.feature_dim,), dtype=np.float32)
    else:
        env.observation_space = Box(low=0.0, high=1.0, shape=(sum(policy_encoder.feature_inner_dim),), dtype=np.float32)
    if params.continuous_action:
        env.action_space = Box(low=env.action_spec[0], high=env.action_spec[1],
                                          shape=env.action_spec[0].shape, dtype=np.float32)
    else:
        env.action_space = Discrete(params.action_dim)

    scripted_policy = get_scripted_policy(env, params)

    policy_algo = training_params.policy_algo
    num_policy_opt_steps = training_params.num_policy_opt_steps
    is_task_learning = False
    policy_params = params.policy_params
    if policy_algo in TASK_LEARNING_ALGO_NAMES:
        pa = policy_params[policy_algo + "_params"]
        policy_kwargs = {"net_arch": pa.fc_dims} # get common parameters here
        gamma = policy_params.discount
        tfb_loc = os.path.join(params.rslts_dir, "tensorboard")
        if policy_params.num_td_steps > 1:
            raise NotImplementedError
        is_task_learning = num_policy_opt_steps > 0
        params.policy_params.batch_size = pa.batch_size
    else:
        num_policy_opt_steps = 0
        params.policy_params.batch_size = 1

    load_policy = params.training_params.load_policy
    if load_policy:
        print(f"loading from {load_policy}")
        policy_name_to_func = {"ppo": PPO, "sac": SAC, "dqn": DQN}
        policy = policy_name_to_func[policy_algo].load(load_policy, env)
    elif policy_algo == "hippo":
        policy = HiPPO(params)
    elif policy_algo == "ppo":
        # We do not include log_std_min and log_std_max, gae is always on
        policy = PPO("MlpPolicy", env, learning_rate=pa.lr, batch_size=pa.batch_size,
                     clip_range=pa.ratio_clip, ent_coef=pa.lambda_entropy,
                     n_steps=pa.n_steps, gae_lambda=pa.lambda_gae_adv, gamma=gamma,
                     policy_kwargs=policy_kwargs, tensorboard_log=tfb_loc, device=device)
    elif policy_algo == "sac":
        policy = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, learning_rate=pa.policy_lr, tau=pa.critic_tau,
                     ent_coef=pa.ent_coef_init, target_entropy=pa.target_ent_coef_final, gamma=gamma,
                     tensorboard_log=tfb_loc, device=device, batch_size=pa.batch_size)
    elif policy_algo == "dqn":
        policy = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, learning_rate=pa.lr, gamma=gamma,
                     tensorboard_log=tfb_loc, device=device, tau=pa.tau, batch_size=pa.batch_size,
                     exploration_initial_eps=pa.exploration_initial_eps, exploration_final_eps=pa.exploration_final_eps)
    elif policy_algo == "random":
        policy = RandomPolicy(env)
    else:
        raise NotImplementedError

    # init replay buffer
    use_prioritized_buffer = replay_buffer_params.prioritized_buffer
    if use_prioritized_buffer:
        replay_buffer = PrioritizedReplayBuffer(params)
    else:
        replay_buffer = ReplayBuffer(params)

    start_step = get_start_step_from_model_loading(params)
    total_step = training_params.total_step
    collect_transitions = training_params.collect_transitions
    train_prop = training_params.train_prop

    if is_task_learning:
        # This initialize the logger
        # We do not use the callback, and we do not maintain a info buffer
        _, callback = policy._setup_learn(
            total_timesteps=total_step,
            tb_log_name=policy_algo,
        )
        policy.num_timesteps = start_step

    # init saving
    writer = None
    if num_dynamics_opt_steps or num_policy_opt_steps:
        writer = SummaryWriter(os.path.join(params.rslts_dir, "tensorboard"))
        model_dir = os.path.join(params.rslts_dir, "trained_models")
        os.makedirs(model_dir, exist_ok=True)
        assert training_params.plot_freq % num_envs == 0
        assert training_params.saving_freq % num_envs == 0

    # init episode variables
    episode_num = start_step * num_envs / env.horizon
    num_dynamics_train_steps = start_step * num_dynamics_opt_steps
    obs = env.reset()
    scripted_policy.reset(obs)

    done = np.zeros(num_envs, dtype=bool)
    previous_done = np.ones(num_envs, dtype=bool)
    success = np.zeros(num_envs, dtype=bool)
    stage_reward = np.zeros(num_envs, dtype=float)
    episode_reward = np.zeros(num_envs)
    episode_step = np.zeros(num_envs)
    is_train = np.random.rand(num_envs) < train_prop
    is_demo = get_is_demo(start_step, params, num_envs)
    ep_info_buffer = deque(maxlen=policy_params.stats_window_size)
    episode_curiosity_reward = np.zeros(num_envs)

    for step in range(start_step, total_step + 1, num_envs):
        is_init_stage = step < training_params.init_step
        # print("{}/{}, init_stage: {}".format(step, total_step, is_init_stage))

        # env interaction and transition saving
        if collect_transitions:
            # reset in the beginning of an episode
            if done.any():
                for i, done_i in enumerate(done):
                    if not done_i:
                        continue

                    if policy_algo == "hippo":
                        policy.reset(i)
                    scripted_policy.reset(obs, i)

                    ep_info_buffer.append([episode_reward[i], episode_curiosity_reward[i], float(success[i]),
                                           stage_reward[i]])
                    if writer is not None and is_task_learning and not is_demo[i]:
                        writer.add_scalar("policy_stat/episode_reward",
                                          safe_mean([ep_info[0] for ep_info in ep_info_buffer]), episode_num)
                        writer.add_scalar("policy_stat/episode_curiosity_reward",
                                          safe_mean([ep_info[1] for ep_info in ep_info_buffer]), episode_num)
                        writer.add_scalar("policy_stat/success",
                                          safe_mean([ep_info[2] for ep_info in ep_info_buffer]), episode_num)
                        writer.add_scalar("policy_stat/stage",
                                          safe_mean([ep_info[3] for ep_info in ep_info_buffer]), episode_num)

                    is_train[i] = np.random.rand() < train_prop
                    is_demo[i] = get_is_demo(step, params)

                    episode_reward[i] = 0
                    episode_step[i] = 0
                    episode_curiosity_reward[i] = 0
                    success[i] = False
                    stage_reward[i] = 0
                    episode_num += 1

            # get action (num_envs, action_dim)
            dynamics.eval()
            if is_task_learning:
                policy.policy.set_training_mode(False)

            if policy_algo == "ppo":
                # TODO: add action clipping? Seems like env can handle it automatically
                obs_feature = obs_dict_to_features(obs, params, device, policy_encoder, tensorize=True)
                with torch.no_grad():
                    actions, values, log_probs = policy.policy(obs_feature)
                action = to_numpy(actions)
            elif is_init_stage:
                action = np.array([policy.action_space.sample() for _ in range(num_envs)])
            elif is_task_learning:
                obs_feature = obs_dict_to_features(obs, params, device, policy_encoder, tensorize=False)
                with torch.no_grad():
                    # TODO: we can add noise to actions if we want
                    action, _ = policy.predict(obs_feature, deterministic=False)
            else:
                action = policy.act(obs)

            if policy_algo != "ppo" and is_demo.any():
                demo_action = scripted_policy.act(obs)
                action[is_demo] = demo_action[is_demo]

            # (num_envs, obs_spec), (num_envs,), (num_envs,), [info] * num_envs
            next_obs, env_reward, done, info = env.step(action)
            policy.num_timesteps += num_envs

            if render:
                env.render()
                time.sleep(0.01)

            if is_task_learning:
                success_step = np.array([info_i["success"] for info_i in info])
                success = success | success_step

            stage_reward = np.array([info_i["stage_completion"] for info_i in info])

            episode_reward += env_reward
            episode_step += 1

            # is_train: if the transition is training data or evaluation data for dynamics_cmi
            replay_buffer.add(obs, action, env_reward, next_obs, done, is_train, info)

            if policy_algo == "ppo":
                if is_init_stage:
                    curiosity_reward = 0
                else:
                    curiosity_reward = dynamics.eval_intrinsic_reward(obs, action, next_obs, done, info)
                episode_curiosity_reward += curiosity_reward
                total_reward = env_reward + curiosity_reward
                policy.rollout_buffer.add(to_numpy(obs_feature), action, total_reward, previous_done, values, log_probs)

            obs = next_obs
            previous_done = done

        # training and logging
        if num_policy_opt_steps and policy_algo == "ppo":
            if policy.rollout_buffer.full:
                next_obs_feature = obs_dict_to_features(next_obs, params, device, policy_encoder, tensorize=True)
                with torch.no_grad():
                    values = policy.policy.predict_values(next_obs_feature)
                policy.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=done)
                policy.logger.dump(step=policy.num_timesteps)
                policy.train()
                policy.rollout_buffer.reset()

        if is_init_stage:
            continue

        loss_details = {"dynamics": [],
                        "dynamics_eval": [],
                        "policy": []}

        if num_dynamics_opt_steps:
            dynamics.train()
            dynamics.setup_annealing(step)
            for _ in range(num_dynamics_opt_steps):
                if not training_params.use_gt_causality:
                    obs_batch, actions_batch, next_obses_batch, idxes_batch = \
                        replay_buffer.sample_dynamics(use_part="train")
                    loss_detail = dynamics.update(obs_batch, actions_batch, next_obses_batch)
                    if use_prioritized_buffer:
                        replay_buffer.update_priorities(idxes_batch, loss_detail["priority"], "dynamics")
                    loss_details["dynamics"].append(loss_detail)
                    num_dynamics_train_steps += 1

                    if num_dynamics_train_steps % training_params.eval_freq == 0:
                        dynamics.eval()
                        obs_batch, actions_batch, next_obses_batch, _ = replay_buffer.sample_dynamics(use_part="eval")
                        loss_detail = dynamics.update(obs_batch, actions_batch, next_obses_batch, eval=True)
                        loss_details["dynamics_eval"].append(loss_detail)
                        dynamics.train()
                else:
                    # We perform update with ground truth causality
                    obs_batch, actions_batch, causal_gt_batch, idxes_batch = \
                        replay_buffer.sample_causal_gt(use_part="train")
                    assert not use_prioritized_buffer
                    loss_detail = dynamics.update(obs_batch, actions_batch, causal_gt_batch)
                    loss_details["dynamics"].append(loss_detail)
                    num_dynamics_train_steps += 1

                    # # Disabled for now: need to fix dictionary bug
                    # if num_dynamics_train_steps % training_params.eval_freq == 0:
                    #     dynamics.eval()
                    #     obs_batch, actions_batch, causal_gt_batch, _ = replay_buffer.sample_dynamics(use_part="eval")
                    #     loss_detail = dynamics.update(obs_batch, actions_batch, causal_gt_batch, eval=True)
                    #     loss_details["dynamics_eval"].append(loss_detail)
                    #     dynamics.train()

        # off-policy RL training part
        if num_policy_opt_steps and policy_algo != "ppo":
            for _ in range(num_policy_opt_steps):
                if policy_algo == "sac" or policy_algo == "dqn":
                    policy.logger.dump(step=policy.num_timesteps)
                    policy.train_with_buffer(replay_buffer, policy_encoder, dynamics)
                else:
                    raise NotImplementedError

        if writer is not None:
            for module_name, module_loss_detail in loss_details.items():
                if not module_loss_detail:
                    continue
                # list of dict to dict of list
                if isinstance(module_loss_detail, list):
                    keys = set().union(*[dic.keys() for dic in module_loss_detail])
                    module_loss_detail = {k: [to_float(dic[k]) for dic in module_loss_detail if k in dic]
                                          for k in keys if k not in ["priority"]}
                for loss_name, loss_values in module_loss_detail.items():
                    writer.add_scalar("{}/{}".format(module_name, loss_name), np.mean(loss_values), step)

            # This can easily get stuck -- not sure why
            if not training_params.use_gt_causality and params.dynamics_params.intrinsic_reward_type != "cai":
                if num_dynamics_opt_steps and step % training_params.plot_freq == 0:
                    plot_adjacency_intervention_mask(params, dynamics, writer, step)

        if step and step % training_params.saving_freq == 0:
            if num_dynamics_opt_steps:
                dynamics.save(os.path.join(model_dir, "dynamics_{}".format(step)))
            if num_policy_opt_steps:
                policy.save(os.path.join(model_dir, "policy_{}".format(step)))
            if collect_transitions:
                if not (is_task_learning and policy_algo == "ppo"):
                    replay_buffer.save()


if __name__ == "__main__":
    params = TrainingParams(training_params_fname="policy_params.json", train=True)
    print("params loaded")
    train(params)
