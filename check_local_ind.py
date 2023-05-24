import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
from joblib import Parallel, delayed

np.set_printoptions(precision=3, suppress=True)

fname = "kitchen_w_skills_test.p"     # clearning_car_test, craft_test
draw_plot = False
threshold_metric = f1_score

position_low = np.array([-0.5, -0.5, 0.7])
position_high = np.array([0.5, 0.5, 1.1])
position_mean = (position_high + position_low) / 2
position_scale = (position_high - position_low) / 2
eef_vel_low = np.array([-2, -2, -2])
eef_vel_high = np.array([2, 2, 2])
eef_vel_mean = (eef_vel_high + eef_vel_low) / 2
eef_vel_scale = (eef_vel_high - eef_vel_low) / 2


def sort_key(e):
    val = e[0]
    if isinstance(val, np.ndarray):
        return -val.sum()
    else:
        return val


def pr_auc_score(y_true, pred):
    precision, recall, thresholds = precision_recall_curve(y_true, pred)
    pr_auc = auc(recall, precision)
    f1 = 2 * precision * recall / (precision + recall).clip(min=1e-10)
    return pr_auc, f1.max(), thresholds[f1.argmax()]


def find_best_threshold_and_result(label, pred):
    prauc, f1, threshold = pr_auc_score(label, pred)
    y_pred = pred >= threshold
    tn, fp, fn, tp = confusion_matrix(label, y_pred, normalize="true").ravel()
    return {"threshold": threshold,
            "accuracy": accuracy_score(label, y_pred),
            "recall": recall_score(label, y_pred),
            "precision": precision_score(label, y_pred),
            "overall_f1": f1,
            "overall_rocauc": roc_auc_score(label, pred),
            "overall_prauc": prauc,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn}


def eval_pred_local_causality():
    expt_evals = {}
    env_name = fname.split("_test")[0]
    rslts = pickle.load(open(fname, "rb"))

    continuous_state = rslts["continuous_state"]

    obs_keys = rslts["dynamics_keys"]
    obs_dims, obj_idxes = [], []
    cum_dim = 0
    for key in obs_keys:
        obs_dim = len(rslts["obs_spec"][key])
        obs_dims.append(obs_dim)
        obj_idxes.append(rslts["obs2obj_index"][cum_dim])
        cum_dim += obs_dim

    obs_keys.append("action")
    obs_dims.append(rslts["action_dim"])
    obj_idxes.append(np.max(obj_idxes) + 1)

    feature_dim = np.sum(obs_dims[:-1])
    action_dim = obs_dims[-1]
    num_objs = np.max(obj_idxes)

    var_idx_2_obs_key_offset = {}
    obs_key_2_var_idx = {}
    obs_key_2_obs_dim = {}
    obj_idx_2_obs_key_var_idx = {}

    total = 0
    for key, dim, obj_idx in zip(obs_keys, obs_dims, obj_idxes):
        obs_key_2_var_idx[key] = (total, total + dim)
        obs_key_2_obs_dim[key] = dim
        for i in range(dim):
            var_idx_2_obs_key_offset[total + i] = (key, i)

        if key.startswith(("robot0_eef", "robot0_gripper")):
            obj_key = key.split("_")[1]
        else:
            obj_key = key.split("_")[0]

        if obj_idx not in obj_idx_2_obs_key_var_idx:
            obj_idx_2_obs_key_var_idx[obj_idx] = [obj_key, total, total + dim]
        else:
            obj_idx_2_obs_key_var_idx[obj_idx][2] = total + dim

        total += dim


    def var_idx_2_pair_key(idx):
        obs_key, offset = var_idx_2_obs_key_offset[idx]
        if obs_key_2_obs_dim[obs_key] == 1:
            return obs_key

        if "pos" in obs_key:
            offset = ["x", "y", "z"][offset]
        elif "quat" in obs_key:
            offset = ["x", "y", "z", "w"][offset]

        return obs_key + "_" + str(offset)


    local_causality = rslts["local_causality"]                      # (num_test_data, feature_dim, feature_dim + action_dim)
    num_test_data = local_causality.shape[0]
    local_causality_obj = np.zeros((num_test_data, feature_dim, num_objs + 1), dtype=bool)
    for i in range(num_objs + 1):
        _, obj_start, obj_end = obj_idx_2_obs_key_var_idx[i]
        local_causality_obj[..., i] = local_causality[..., obj_start:obj_end].any(axis=-1)

    for path in rslts:
        if path in ["obs", "action", "next_obs", "local_causality",
                    "continuous_state", "dynamics_keys", "obs_spec", "action_dim", "obs2obj_index"]:
            continue

        print(path)
        path_rslts = rslts[path]
        expt = path.split("_2023")[0]
        expt_evals[expt] = expt_eval = {}

        feature = path_rslts["feature"]                             # (num_test_data, feature_dim)
        next_feature = path_rslts["next_feature"]                   # (num_test_data, feature_dim)
        pred_next_feature = path_rslts["pred_next_feature"][:, 0]   # (num_test_data, feature_dim)

        if continuous_state:
            expt_eval["prediction_error"] = prediction_error = {}
            diff = np.abs(pred_next_feature - next_feature)
            for obs_key in obs_keys:
                obs_start, obs_end = obs_key_2_var_idx[obs_key]
                obs_i_diff = diff[:, obs_start:obs_end]
                # if obs_key == "eef" or obs_key.endswith("pos"):
                #     obs_i_diff = obs_i_diff * position_scale
                # elif obs_key == "eef_vel":
                #     obs_i_diff = obs_i_diff * eef_vel_scale

                obs_pred_diff = obs_i_diff.sum(axis=-1)
                prediction_error[obs_key] = (obs_pred_diff.mean(axis=0), obs_pred_diff.std(axis=0) / len(obs_pred_diff))
        else:
            expt_eval["prediction_acc"] = prediction_acc = {}
            expt_eval["prediction_change_acc"] = prediction_change_acc = {}
            acc = pred_next_feature == next_feature
            change = feature != next_feature
            for obs_key in obs_keys:
                obs_start, obs_end = obs_key_2_var_idx[obs_key]
                obs_i_acc = acc[:, obs_start:obs_end]
                prediction_acc[obs_key] = obs_i_acc.mean(axis=0)
                change_mask = change[:, obs_start:obs_end].any(axis=-1)
                if change_mask.any():
                    prediction_change_acc[obs_key] = obs_i_acc[change_mask].mean(axis=0)

        plot_dir = "{}_mask_plots".format(env_name)
        os.makedirs(plot_dir, exist_ok=True)

        for mask_key, mask in path_rslts.items():
            if mask_key not in ["grad_mask", "sample_cmi_mask", "attn_mask"] or mask is None:
                continue

            if not np.isfinite(mask).all():
                continue

            num_parents = mask.shape[-1]
            assert num_parents in [feature_dim + action_dim, num_objs + 1]
            is_parent_obj = (num_parents == num_objs + 1)

            label = local_causality_obj if is_parent_obj else local_causality

            label_flat = label.flatten()
            mask_flat = mask.flatten()

            expt_eval[mask_key] = mask_metrics = find_best_threshold_and_result(label_flat, mask_flat)
            mask_metrics["rocauc"] = mask_rocauc = {}
            mask_metrics["prauc"] = mask_prauc = {}
            mask_metrics["f1"] = mask_f1 = {}
            threshold = expt_eval[mask_key]["threshold"]

            if "grad" in mask_key:
                mask_flat = np.log10(np.maximum(mask_flat, np.finfo(np.float32).tiny))
                threshold = np.log10(np.maximum(threshold, np.finfo(np.float32).tiny))

            plt.figure(figsize=(4, 3))
            plt.hist(mask_flat[label_flat], bins=30, density=True, label="dep", alpha=0.7)
            plt.hist(mask_flat[~label_flat], bins=30, density=True, label="indep", alpha=0.7)
            ymin, ymax = plt.ylim()
            plt.vlines(threshold, ymin, ymax)
            plt.ylim(ymin, ymax)
            plt.title("roc auc: {:.3f}, pr auc: {:.3f}".format(mask_metrics["overall_rocauc"],
                                                               mask_metrics["overall_prauc"]))
            plt.xlabel("mask val")
            plt.legend()
            plt.ylabel("density")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "overall_{}_{}".format(expt, mask_key)), dpi=200)

            for i in range(feature_dim):
                pred_key = var_idx_2_pair_key(i)
                for j in range(num_parents):

                    if is_parent_obj:
                        input_key, obj_start, obj_end = obj_idx_2_obs_key_var_idx[j]
                        label = local_causality_obj[:, i, j]
                    else:
                        input_key = var_idx_2_pair_key(j)
                        label = local_causality[:, i, j]

                    mask_val = mask[:, 0, i, j]

                    if (label == label[0]).all():
                        continue
                        rocauc = prauc = 1
                    else:
                        rocauc = roc_auc_score(label, mask_val)
                        prauc, f1, pair_threshold = pr_auc_score(label, mask_val)

                    pair_name = input_key + " -> " + pred_key
                    mask_rocauc[pair_name] = rocauc
                    mask_prauc[pair_name] = prauc
                    mask_f1[pair_name] = f1

                    if not draw_plot:
                        continue

                    dep_mask_val = mask_val[label]
                    indep_mask_val = mask_val[~label]
                    if "grad" in mask_key:
                        dep_mask_val = np.log10(np.maximum(dep_mask_val, np.finfo(np.float32).tiny))
                        indep_mask_val = np.log10(np.maximum(indep_mask_val, np.finfo(np.float32).tiny))
                        pair_threshold = np.log10(np.maximum(pair_threshold, np.finfo(np.float32).tiny))

                    plt.figure(figsize=(4, 3))
                    if label.any():
                        plt.hist(dep_mask_val, bins=10, density=True, label="dep", alpha=0.7)
                    if (~label).any():
                        plt.hist(indep_mask_val, bins=10, density=True, label="indep", alpha=0.7)
                    ymin, ymax = plt.ylim()
                    plt.vlines(threshold, ymin, ymax, color="red", label="overall thres")
                    plt.vlines(pair_threshold, ymin, ymax, color="green", label="pair thres")
                    plt.ylim(ymin, ymax)
                    plt.title("roc auc: {:.3f}, pr auc: {:.3f}, f1: {:.3f}".format(rocauc, prauc, f1))
                    plt.xlabel("mask val")
                    plt.legend()
                    plt.ylabel("density")
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, "{}_{}_{}".format(pair_name, expt, mask_key)), dpi=200)

                plt.close("all")

    # prediction error
    for obs_key in obs_keys[:-1]:
        if continuous_state:
            pred_scores = [(expt_eval["prediction_error"][obs_key], expt)
                           for expt, expt_eval in expt_evals.items()]
        else:
            pred_scores = [(expt_eval["prediction_acc"][obs_key], expt_eval["prediction_change_acc"].get(obs_key, None), expt)
                           for expt, expt_eval in expt_evals.items()]
        pred_scores.sort(key=sort_key)
        print(obs_key)
        for pred_score in pred_scores:
            print(*pred_score)
        print()

    # local causality quality
    local_causality_rslts = {}
    for expt, expt_eval in expt_evals.items():
        for mask_key, mask_metrics in expt_eval.items():
            if "mask" not in mask_key:
                continue
            for metric_name, performance in mask_metrics.items():
                if metric_name not in local_causality_rslts:
                    local_causality_rslts[metric_name] = []
                if metric_name in ["rocauc", "prauc", "f1"]:
                    performance = np.mean(list(performance.values()))
                local_causality_rslts[metric_name].append([performance, expt, mask_key])

    for metric_name, masks_performance in local_causality_rslts.items():
        print(metric_name)
        masks_performance.sort(key=sort_key)
        for metric_score, expt, mask_key in masks_performance:
            if metric_name == "threshold":
                print("{}, {}, {}".format(metric_score, expt, mask_key))
            else:
                print("{:.5f}, {}, {}".format(metric_score, expt, mask_key))
        print()

    return expt_evals


if __name__ == "__main__":
    eval_pred_local_causality()
