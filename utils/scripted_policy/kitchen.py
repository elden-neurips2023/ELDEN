import random
import numpy as np


class ScriptedKitchen:
    def __init__(self, env, params):
        self.params = params

        scripted_policy_params = params.scripted_policy_params
        self.random_action_prob = scripted_policy_params.random_action_prob
        self.action_noise_scale = scripted_policy_params.action_noise_scale
        self.drop_prob = scripted_policy_params.drop_prob
        self.action_scale = 20.0
        self.hover_height = 0.95
        self.pot_hover_height = 0.85
        self.bread_drop_height = 0.90
        self.pot_to_stove_drop_height = 0.85
        self.pot_to_target_drop_height = 0.85
        self.button_push_offset = 0.05

        self.task = self.params.env_params.env_name

        self.action_low, self.action_high = params.action_spec

        global_low, global_high = params.normalization_range
        global_low = np.array(global_low)
        global_high = np.array(global_high)
        self.global_mean = (global_high + global_low) / 2
        self.global_scale = (global_high - global_low) / 2

        self.STATES = ["BREAD_PICKING", "BREAD_MOVING", "POT_PICKING", "POT_MOVING",
                       "BUTTON_TURNING_ON", "DISH_PICKING", "DISH_MOVING", "BUTTON_TURNING_OFF", "DONE"]

    def reset(self, obs):
        self.state = random.choice(["BREAD_PICKING", "BUTTON_TURNING_ON"])
        self.pot_bread_on_stove = False
        self.reach_button_push_start = False

    def grasping_act(self, eef_pos, object_pos, hover_height=None):
        action = np.zeros_like(self.action_low)
        action[-1] = -np.random.rand()

        placement = object_pos - eef_pos
        xy_place, z_place = placement[:2], placement[-1]

        if hover_height is None:
            hover_height = self.hover_height

        action_noise = 0
        if np.linalg.norm(xy_place) >= 0.02:
            if eef_pos[2] < hover_height:
                action[2] = hover_height + 0.05 - eef_pos[2]
            else:
                action[:2] = xy_place
            action_noise = 0.1
            action[-1] = -np.random.rand()
        elif np.linalg.norm(placement) >= 0.01:
            action[:3] = placement
        else:
            action[:3] = placement
            action[-1] = np.random.uniform(0.5, 1)

        action[:3] *= self.action_scale

        return action, action_noise

    def reaching_act(self, eef_pos, goal_pos, hover=True):
        action = np.zeros_like(self.action_low)
        action[-1] = np.random.rand()

        placement = goal_pos - eef_pos
        xy_place, z_place = placement[:2], placement[-1]

        if np.linalg.norm(xy_place) >= 0.02:
            if hover and eef_pos[2] < self.hover_height:
                action[2] = self.hover_height + 0.05 - eef_pos[2]
                action[-1] = -np.random.rand()
            else:
                action[:2] = xy_place
        else:
            action[:3] = placement

        action[:3] *= self.action_scale

        action_noise = 0.1

        return action, action_noise

    def act(self, obs):
        eef_pos = obs["robot0_eef_pos"] * self.global_scale + self.global_mean
        bread_pos = obs["bread_pos"] * self.global_scale + self.global_mean
        pot_pos = obs["pot_pos"] * self.global_scale + self.global_mean
        stove_xy = (obs["stove_pos"] * self.global_scale + self.global_mean)[:2]
        target_xy = (obs["target_pos"] * self.global_scale + self.global_mean)[:2]

        button_handle_pos = obs["button_handle_pos"] * self.global_scale + self.global_mean
        pot_handle_pos = obs["pot_handle_pos"] * self.global_scale + self.global_mean
        bread_cooked = obs["bread_cooked"][0]
        button_joint_qpos = obs["button_joint_qpos"][0]

        bread_grasped = obs["bread_grasped"][0]
        pot_grasped = obs["pot_grasped"][0]
        pot_touched = obs["pot_touched"][0]

        bread_cooked = bread_cooked == 1
        button_on = button_joint_qpos > 0
        bread_close_to_pot = np.linalg.norm(bread_pos[:2] - pot_pos[:2]) < 0.02
        pot_on_stove = np.linalg.norm(pot_pos[:2] - stove_xy) < 0.05
        pot_on_target = np.linalg.norm(pot_pos[:2] - target_xy) < 0.05

        prev_state = self.state
        assert self.state in self.STATES
        # determine current state
        if self.state == "BREAD_PICKING":
            if bread_grasped:
                self.state = "BREAD_MOVING"
        elif self.state == "BREAD_MOVING":
            if not bread_grasped:
                if bread_close_to_pot:
                    self.state = "POT_PICKING"
                else:
                    self.state = "BREAD_PICKING"
        elif self.state == "POT_PICKING":
            if pot_grasped:
                self.state = "POT_MOVING"
        elif self.state == "POT_MOVING":
            if not pot_grasped:
                if pot_on_stove:
                    self.pot_bread_on_stove = True
                    if not pot_touched:
                        if not button_on:
                            self.state = "BUTTON_TURNING_ON"
                        elif bread_cooked:
                            self.state = "DISH_PICKING"
                else:
                    self.state = "POT_PICKING"
        elif self.state == "BUTTON_TURNING_ON":
            if button_on:
                if not self.pot_bread_on_stove:
                    self.state = "BREAD_PICKING"
                elif bread_cooked:
                    self.state = "DISH_PICKING"
        elif self.state == "DISH_PICKING":
            if pot_grasped:
                self.state = "DISH_MOVING"
        elif self.state == "DISH_MOVING":
            if not pot_grasped:
                if pot_on_target:
                    if not pot_touched:
                        self.state = "BUTTON_TURNING_OFF"
                else:
                    self.state = "DISH_PICKING"
        elif self.state == "BUTTON_TURNING_OFF":
            if not button_on:
                self.state = "DONE"

        if prev_state != self.state and self.state in ["BUTTON_TURNING_ON", "BUTTON_TURNING_OFF"]:
            self.reach_button_push_start = False

        if np.random.rand() < self.random_action_prob:
            return self.act_randomly()

        action = np.zeros_like(self.action_low)
        action[-1] = -np.random.rand()
        action_noise = 0.1

        if self.state == "BREAD_PICKING":
            action, action_noise = self.grasping_act(eef_pos, bread_pos)
        elif self.state == "BREAD_MOVING":
            bread_goal_pos = pot_pos.copy()
            bread_goal_pos[2] = self.bread_drop_height
            action, action_noise = self.grasping_act(bread_pos, bread_goal_pos)
            if bread_close_to_pot:
                action[-1] = -np.random.rand()
            else:
                action[-1] = np.random.rand()
        elif self.state == "POT_PICKING":
            action, action_noise = self.grasping_act(eef_pos, pot_handle_pos)
        elif self.state == "POT_MOVING":
            pot_goal_pos = np.concatenate([stove_xy, [self.pot_to_stove_drop_height]])
            action, action_noise = self.grasping_act(pot_pos, pot_goal_pos, hover_height=self.pot_hover_height)
            if np.linalg.norm(pot_pos - pot_goal_pos) < 0.02:
                action[:-1] = 0
                action[-1] = -np.random.rand()
            else:
                action[-1] = np.random.uniform(0.5, 1)
        elif self.state == "BUTTON_TURNING_ON":
            if not self.reach_button_push_start:
                start_offset = np.array([0, -self.button_push_offset, 0])
                action, action_noise = self.reaching_act(eef_pos, button_handle_pos + start_offset)
                if np.linalg.norm(eef_pos - (button_handle_pos + start_offset)) < 0.01:
                    self.reach_button_push_start = True
            else:
                end_offset = np.array([0, self.button_push_offset, 0])
                action, action_noise = self.reaching_act(eef_pos, button_handle_pos + end_offset, hover=False)
        elif self.state == "DISH_PICKING":
            action, action_noise = self.grasping_act(eef_pos, pot_handle_pos)
        elif self.state == "DISH_MOVING":
            pot_goal_pos = np.concatenate([target_xy, [self.pot_to_target_drop_height]])
            action, action_noise = self.grasping_act(pot_pos, pot_goal_pos, hover_height=self.pot_hover_height)
            if np.linalg.norm(pot_pos - pot_goal_pos) < 0.02:
                action[:-1] = 0
                action[-1] = -np.random.rand()
            else:
                action[-1] = np.random.uniform(0.5, 1)
        elif self.state == "BUTTON_TURNING_OFF":
            if not self.reach_button_push_start:
                start_offset = np.array([0, self.button_push_offset, 0])
                action, action_noise = self.reaching_act(eef_pos, button_handle_pos + start_offset)
                if np.linalg.norm(eef_pos - (button_handle_pos + start_offset)) < 0.01:
                    self.reach_button_push_start = True
            else:
                end_offset = np.array([0, -self.button_push_offset, 0])
                action, action_noise = self.reaching_act(eef_pos, button_handle_pos + end_offset, hover=False)

        action = np.clip(action, self.action_low * 0.9, self.action_high * 0.9)
        action[:3] += np.random.uniform(-action_noise, action_noise, 3)

        return action

    def act_randomly(self):
        return np.random.uniform(self.action_low, self.action_high)
