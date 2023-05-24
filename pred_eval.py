import os
import pickle
import numpy as np

import torch
np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)


from model.encoder import IdentityEncoder, ObjectEncoder
from model.ensemble_dynamics_grad_attn import EnsembleDynamicsGradAttn

from utils.utils import TrainingParams, update_obs_act_spec, set_seed_everywhere, get_env, to_numpy
from utils.scripted_policy.misc import get_scripted_policy
from utils.eval_utils import get_noisy_obs, get_random_obs, kl_div_mixture_app
import torch.nn.functional as F


if __name__ == "__main__":
    repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    load_paths = ["kitchen_w_skills_none_mask_2023_05_12_22_59_14",
                  "kitchen_w_skills_sample_mask_2023_05_12_23_00_15",
                  "kitchen_w_skills_none_mask_reg_1en4_2023_05_13_03_53_09",
                  "kitchen_w_skills_none_mask_reg_1en3_2023_05_13_03_53_45",
                  "kitchen_w_skills_none_mask_reg_1en2_2023_05_13_03_54_19",]

    # load_paths = ["clearning_car_none_mask_mixup_01_2023_05_06_20_19_45",
    #               "clearning_car_none_mask_mixup_2_2023_05_06_20_21_31",
    #               "clearning_car_sample_mask_mixup_01_2023_05_06_20_24_50",
    #               "clearning_car_sample_mask_mixup_2_2023_05_06_20_25_06",]

    save_fname = "test.p"
    num_epi = 50
    num_pred_steps = 1
    batch_size = 32
    model_step = None           # 1800000

    cuda_id = 0
    seed = 0

    # ======================================= params to overwrite begins ======================================= #
    # contrastive params
    num_pred_samples = 4096
    num_pred_iters = 1
    pred_iter_sigma_init = 0.33
    pred_iter_sigma_shrink = 0.5

    # ======================================== params to overwrite ends ======================================== #

    device = torch.device("cuda:{}".format(cuda_id) if torch.cuda.is_available() else "cpu")
    print(f"using device:{device}")

    # get test transitions
    load_path = os.path.join(repo_path, "interesting_models", load_paths[0])

    params_fname = os.path.join(load_path, "params.json")
    params = TrainingParams(training_params_fname=params_fname, train=False)

    print("loaded training parameters")

    params.seed = seed
    env_params = params.env_params
    env_params.num_envs = 1
    save_fname = env_params.env_name + "_" + save_fname
    set_seed_everywhere(seed)
    params.device = device

    env = get_env(params)

    update_obs_act_spec(env, params)

    policy = get_scripted_policy(env, params)

    obs_buffer, action_buffer, next_obs_buffer, local_causality_buffer = [], [], [], []
    if num_pred_steps != 1:
        raise NotImplementedError

    for e in range(num_epi):
        print("{}/{}".format(e + 1, num_epi))
        done = False

        obs = env.reset()
        policy.reset(obs)

        step = 0

        while not done:
            step += 1
            action = policy.act(obs)
            next_obs, reward, done, info = env.step(action)
            done = done[0]

            if not done:
                obs_buffer.append(obs)
                action_buffer.append(action)
                next_obs_buffer.append(next_obs)
                local_causality_buffer.append(info[0]["local_causality"])

            obs = next_obs

    obs_buffer = {k: np.concatenate([obs[k] for obs in obs_buffer], axis=0)
                  for k in obs_buffer[0].keys()}
    action_buffer = np.concatenate(action_buffer, axis=0)
    next_obs_buffer = {k: np.concatenate([next_obs[k] for next_obs in next_obs_buffer], axis=0)
                       for k in next_obs_buffer[0].keys()}
    local_causality_buffer = np.stack(local_causality_buffer, axis=0)

    num_test_data = action_buffer.shape[0]
    print("number of test transitions:", num_test_data)

    # eval prediction
    performances = {"obs": obs_buffer,
                    "action": action_buffer,
                    "next_obs": next_obs_buffer,
                    "local_causality": local_causality_buffer}

    obj_encoder = ObjectEncoder(params)
    performances.update({"continuous_state": params.continuous_state,
                         "dynamics_keys": params.dynamics_keys,
                         "obs_spec": params.obs_spec,
                         "action_dim": params.action_dim if params.continuous_action else 1,
                         "obs2obj_index": obj_encoder.obs2obj_index})

    for path in load_paths:
        print("test", path)
        load_path = os.path.join(repo_path, "interesting_models", path)

        params_fname = os.path.join(load_path, "params.json")
        params = TrainingParams(training_params_fname=params_fname, train=False)

        if model_step is None:
            dynamics_fnames = [fname for fname in os.listdir(load_path) if "dynamics" in fname]
            assert len(dynamics_fnames) == 1
            dynamics_fname = dynamics_fnames[0]
            dynamics_fname = os.path.join(load_path, dynamics_fname)
        else:
            dynamics_fname = os.path.join(load_path, "dynamics_" + str(model_step))
            if not os.path.exists(dynamics_fname):
                print("warning:", dynamics_fname, "doesn't exist")
                continue

        params.seed = seed
        params.training_params.load_dynamics = dynamics_fname
        env_params = params.env_params

        set_seed_everywhere(seed)
        params.device = device
        training_params = params.training_params

        update_obs_act_spec(env, params)
        encoder = IdentityEncoder(params, key_type="dynamics_keys")
        obj_encoder = ObjectEncoder(params)
        params.feature_dim = encoder.feature_dim

        params.dynamics_params.contrastive_params.num_pred_samples = num_pred_samples
        params.dynamics_params.contrastive_params.num_pred_iters = num_pred_iters
        params.dynamics_params.contrastive_params.pred_iter_sigma_init = pred_iter_sigma_init
        params.dynamics_params.contrastive_params.pred_iter_sigma_shrink = pred_iter_sigma_shrink

        dynamics_algo = params.training_params.dynamics_algo
        if dynamics_algo == "grad_attn":
            Dynamics = EnsembleDynamicsGradAttn
        else:
            raise NotImplementedError
        dynamics = Dynamics(encoder, obj_encoder, params)
        dynamics.eval()

        feature_buffer = []
        next_feature_buffer = []
        pred_next_feature_buffer = []
        mask_buffer = []
        for i in range(0, num_test_data, batch_size):
            print("{}/{}".format(i + batch_size, num_test_data))
            obs = {k: v[i:i + batch_size] for k, v in obs_buffer.items()}
            action = action_buffer[i:i + batch_size]
            next_obs = {k: v[i:i + batch_size, :] for k, v in next_obs_buffer.items()}
            feature, next_features, pred_next_features, masks = dynamics.eval_prediction(obs, action, next_obs)

            feature_buffer.append(to_numpy(feature))
            next_feature_buffer.append(to_numpy(next_features))
            pred_next_feature_buffer.append(to_numpy(pred_next_features))
            mask_buffer.append(to_numpy(masks))

        feature = np.concatenate(feature_buffer, axis=0)                        # (num_test_data, feature_dim)
        next_feature = np.concatenate(next_feature_buffer, axis=0)              # (num_test_data, feature_dim)
        pred_next_feature = np.concatenate(pred_next_feature_buffer, axis=0)    # (num_test_data, feature_dim)
        # list of dict to dict of list
        masks = {mask_key: None if mask_val is None else
                 np.concatenate([masks[mask_key] for masks in mask_buffer], axis=0)
                 for mask_key, mask_val in mask_buffer[0].items()}

        performance = masks
        performance.update({"feature": feature,
                            "next_feature": next_feature,
                            "pred_next_feature": pred_next_feature})
        performances[path] = performance

    with open(save_fname, "wb") as f:
        pickle.dump(performances, f)
