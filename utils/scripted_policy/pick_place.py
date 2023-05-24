import random
import numpy as np


class ScriptedPickAndPlace:
    def __init__(self, env, params):
        self.params = params

        block_env_params = params.env_params.manipulation_env_params.block_env_params
        self.obj_to_pick_names = ["mov" + str(i) for i in range(block_env_params.num_movable_objects)]

        scripted_policy_params = params.scripted_policy_params
        self.random_action_prob = scripted_policy_params.random_action_prob
        self.action_noise_scale = scripted_policy_params.action_noise_scale
        self.drop_prob = scripted_policy_params.drop_prob

        pick_place_params = scripted_policy_params.pick_place_params
        self.push_prob = pick_place_params.push_prob
        self.push_z = pick_place_params.push_z
        self.rough_grasp_prob = pick_place_params.rough_grasp_prob
        self.rough_grasp_noise_scale = pick_place_params.rough_grasp_noise_scale
        self.rough_move_prob = pick_place_params.rough_move_prob

        self.action_low, self.action_high = params.action_spec

        self.is_stack = False
        self.is_demo = pick_place_params.is_demo
        if self.is_demo:
            self.is_stack = self.params.env_params.env_name == "BlockStack"
            self.drop_prob = 0.0
            self.action_noise_scale = 0.0
            self.push_prob = 0.0
            self.rough_grasp_prob = 0.0
            self.rough_grasp_noise_scale = 0.0
            self.rough_move_prob = 0.0

        self.action_scale = 20.0
        self.workspace_low, self.workspace_high = [-0.3, -0.4, 0.82], [0.3, 0.4, 1.00]

        self.episode_params_inited = False

    def reset(self, obs):
        self.obj_to_pick_name = random.choice(self.obj_to_pick_names)

        self.release = False
        self.release_step = 0
        self.push = np.random.rand() < self.push_prob

        self.rough_grasp = np.random.rand() < self.rough_grasp_prob
        self.rough_move = np.random.rand() < self.rough_move_prob
        self.goal = self.sample_goal(obs)

    def sample_goal(self, obs):
        if self.is_demo:
            if self.is_stack:
                goal = obs["unmov0_pos"].copy()
                goal[-1] = 0.95
                return goal
            else:
                return obs["goal_pos"]

        goal_key = "goal_" + self.obj_to_pick_name + "_pos"
        if goal_key in obs:
            return obs[goal_key]

        goal = np.random.uniform(self.workspace_low, self.workspace_high)
        if self.push:
            goal[2] = self.push_z

        # print("\nnew goal: {}\n".format(goal))

        return goal

    def act(self, obs):
        if np.random.rand() < self.random_action_prob:
            return self.act_randomly()

        low, high = self.params.action_spec
        eef_pos = obs["robot0_eef_pos"]
        object_pos = obs[self.obj_to_pick_name + "_pos"]
        grasped = obs[self.obj_to_pick_name + "_grasped"]

        action = np.zeros_like(low)
        action[-1] = -1

        if not grasped:
            placement = object_pos - eef_pos
            xy_place, z_place = placement[:2], placement[-1]

            if np.abs(z_place) >= 0.1:
                action[:3] = placement
                action[-1] = np.random.rand() * 2 - 1
                action_noise_scale = self.rough_grasp_noise_scale
            else:
                if self.rough_grasp:
                    action[:3] = placement
                    action[-1] = np.random.rand() * 2.3 - 1.3
                    action_noise_scale = 0.2
                else:
                    if np.linalg.norm(xy_place) >= 0.02:
                        action[:2] = xy_place
                    elif np.abs(z_place) >= 0.02:
                        action[2] = z_place
                    elif np.linalg.norm(placement) >= 0.01:
                        action[:3] = placement
                    else:
                        action[-1] = 1
                        # print("try to grasp, success:", bool(grasped))
                    action_noise_scale = 0.0

        else:
            # eef position
            action_noise_scale = self.action_noise_scale

            placement = self.goal - eef_pos
            if np.linalg.norm(placement) < 0.1:
                self.goal = self.sample_goal(obs)

            if self.rough_move:
                action = self.act_randomly()
            else:
                action[:3] = placement

            # gripper
            to_release = np.random.rand() < self.drop_prob
            if self.is_stack and np.linalg.norm(placement) < 0.02:
                to_release = True
            self.release = self.release or to_release
            if self.release:
                action[-1] = np.random.random() * 2 - 1
                self.release_step += 1
                if self.release_step >= 3:
                    self.release = False
                    self.release_step = 0
            else:
                action[-1] = np.random.random()

        action[:3] *= self.action_scale
        noise = np.random.uniform(low=-action_noise_scale, high=action_noise_scale, size=3)
        if self.push:
            noise[2] = np.clip(noise[2], -np.inf, 0.02)
        action[:3] += noise

        action[:3] = np.clip(action, low, high)[:3]

        return action

    def act_randomly(self):
        return np.random.uniform(self.action_low, self.action_high)
