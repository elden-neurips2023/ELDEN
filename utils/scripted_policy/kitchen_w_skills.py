import random
import numpy as np


class ScriptedKitchenWithSkills:
    def __init__(self, env, params):
        self.env = env
        self.params = params

        scripted_policy_params = params.scripted_policy_params
        self.random_action_prob = scripted_policy_params.random_action_prob
        self.action_dim = params.action_dim

        global_low, global_high = params.normalization_range
        global_low, global_high = np.array(global_low), np.array(global_high)
        self.global_mean = (global_high + global_low) / 2
        self.global_scale = (global_high - global_low) / 2

        self.action_names = env.action_names
        self.action_dict = {action_name: i for i, action_name in enumerate(self.action_names)}

        self.STATES = ["BUTTER_PICKING", "BUTTER_MOVING", 
                       "POT_PICKING", "POT_MOVING",
                       "MEATBALL_PICKING", "MEATBALL_MOVING",
                       "BUTTON_TURNING_ON",
                       "DISH_PICKING", "DISH_MOVING",
                       "BUTTON_TURNING_OFF",
                       "DONE"]

    def reset(self, obs):
        self.state = random.choice(["BUTTER_PICKING", "POT_PICKING", "BUTTON_TURNING_ON"])

    def act(self, obs):
        eef_pos = obs["robot0_eef_pos"] * self.global_scale + self.global_mean
        butter_pos = obs["butter_pos"] * self.global_scale + self.global_mean
        meatball_pos = obs["meatball_pos"] * self.global_scale + self.global_mean
        pot_pos = obs["pot_pos"] * self.global_scale + self.global_mean
        pot_handle_pos = obs["pot_handle_pos"] * self.global_scale + self.global_mean
        button_handle_pos = obs["button_handle_pos"] * self.global_scale + self.global_mean

        button_pos = obs["button_pos"] * self.global_scale + self.global_mean
        stove_pos = obs["stove_pos"] * self.global_scale + self.global_mean
        target_pos = obs["target_pos"] * self.global_scale + self.global_mean

        butter_melt_status = obs["butter_melt_status"][0]
        meatball_overcooked = obs["meatball_overcooked"][0]
        meatball_cook_status = obs["meatball_cook_status"][0]
        butter_melted = butter_melt_status == 1
        meatball_cooked = meatball_cook_status == 1

        butter_in_pot = obs["butter_in_pot"][0]
        if butter_melt_status == 1:
            butter_in_pot = True
        meatball_in_pot = obs["meatball_in_pot"][0]

        butter_grasped = obs["butter_grasped"][0]
        butter_touched = obs["butter_touched"][0]
        meatball_grasped = obs["meatball_grasped"][0]
        pot_grasped = obs["pot_grasped"][0]
        pot_touched = obs["pot_touched"][0]
        button_joint_qpos = obs["button_joint_qpos"][0]

        button_on = button_joint_qpos > 0

        eef_close_to_butter = np.linalg.norm(eef_pos - butter_pos) < 0.02
        eef_close_to_meatball = np.linalg.norm(eef_pos - meatball_pos) < 0.02
        eef_close_to_pot_handle = np.linalg.norm(eef_pos - pot_handle_pos) < 0.02
        eef_close_to_button = np.linalg.norm(eef_pos[:2] - button_pos[:2]) < 0.02

        butter_close_to_pot = np.linalg.norm(butter_pos[:2] - pot_pos[:2]) < 0.075
        meatball_close_to_pot = np.linalg.norm(meatball_pos[:2] - pot_pos[:2]) < 0.075
        pot_close_to_stove = np.linalg.norm(pot_pos - stove_pos) < 0.075
        pot_close_to_target = np.linalg.norm(pot_pos - target_pos) < 0.075

        pot_on_stove = pot_close_to_stove and not pot_touched
        pot_on_target = pot_close_to_target and not pot_touched

        # determine current state
        if self.state == "BUTTER_PICKING":
            if butter_grasped:
                self.state = "BUTTER_MOVING"
        elif self.state == "BUTTER_MOVING":
            if not butter_grasped:
                if butter_close_to_pot:
                    if not pot_on_stove:
                        self.state = "POT_PICKING"
                    elif not button_on:
                        self.state = "BUTTON_TURNING_ON"
                    else:
                        self.state = "POT_MOVING"
                else:
                    self.state = "BUTTER_PICKING"
        elif self.state == "POT_PICKING":
            if pot_grasped:
                self.state = "POT_MOVING"
        elif self.state == "POT_MOVING":
            if not pot_grasped:
                if pot_close_to_stove:
                    if not (pot_touched and eef_close_to_pot_handle):
                        if not button_on:
                            self.state = "BUTTON_TURNING_ON"
                        elif not butter_melted and not butter_in_pot:
                            self.state = "BUTTER_PICKING"
                        elif butter_melted and not meatball_in_pot:
                            self.state = "MEATBALL_PICKING"
                        elif meatball_cooked:
                            self.state = "DISH_PICKING"
                        else:
                            self.state = "POT_MOVING"
                else:
                    self.state = "POT_PICKING"
        elif self.state == "BUTTON_TURNING_ON":
            if button_on:
                if not pot_on_stove:
                    self.state = "POT_PICKING"
                elif not butter_in_pot:
                    self.state = "BUTTER_PICKING"
                elif butter_melted and not meatball_in_pot:
                    self.state = "MEATBALL_PICKING"
                elif meatball_cooked:
                    self.state = "DISH_PICKING"
                else:
                    self.state = "POT_MOVING"
        elif self.state == "MEATBALL_PICKING":
            if meatball_grasped:
                self.state = "MEATBALL_MOVING"
        elif self.state == "MEATBALL_MOVING":
            if not meatball_grasped:
                if meatball_in_pot:
                    if pot_on_stove:
                        self.state = "POT_MOVING"
                    else:
                        self.state = "POT_PICKING"
                else:
                    self.state = "MEATBALL_PICKING"
        elif self.state == "DISH_PICKING":
            if pot_grasped:
                self.state = "DISH_MOVING"
        elif self.state == "DISH_MOVING":
            if not pot_grasped:
                if pot_close_to_target:
                    if not (pot_touched and eef_close_to_pot_handle):
                        self.state = "BUTTON_TURNING_OFF"
                else:
                    self.state = "DISH_PICKING"
        elif self.state == "BUTTON_TURNING_OFF":
            if not button_on:
                self.state = "DONE"
        assert self.state in self.STATES

        action = self.act_randomly()
        if np.random.rand() < self.random_action_prob:
            return action

        if self.state == "BUTTER_PICKING":
            if eef_close_to_butter:
                action = "GRASP_BUTTER"
            else:
                action = "MOVE_TO_BUTTER"
        elif self.state == "BUTTER_MOVING":
            if butter_close_to_pot:
                action = "DROP"
            else:
                action = "MOVE_TO_POT"
        elif self.state == "POT_PICKING":
            if eef_close_to_pot_handle:
                action = "GRASP_POT_HANDLE"
            else:
                action = "MOVE_TO_POT_HANDLE"
        elif self.state == "POT_MOVING":
            if pot_close_to_stove:
                if pot_touched and eef_close_to_pot_handle:
                    action = "DROP"
            else:
                action = "MOVE_TO_STOVE"
        elif self.state == "BUTTON_TURNING_ON":
            if not button_on:
                if eef_close_to_button:
                    action = "TOGGLE"
                else:
                    action = "MOVE_TO_BUTTON"
        elif self.state == "MEATBALL_PICKING":
            if eef_close_to_meatball:
                action = "GRASP_MEATBALL"
            else:
                action = "MOVE_TO_MEATBALL"
        elif self.state == "MEATBALL_MOVING":
            if meatball_close_to_pot:
                action = "DROP"
            else:
                action = "MOVE_TO_POT"
        elif self.state == "DISH_PICKING":
            if eef_close_to_pot_handle:
                action = "GRASP_POT_HANDLE"
            else:
                action = "MOVE_TO_POT_HANDLE"
        elif self.state == "DISH_MOVING":
            if pot_close_to_target:
                action = "DROP"
            else:
                action = "MOVE_TO_TARGET"
        elif self.state == "BUTTON_TURNING_OFF":
            if button_on:
                if eef_close_to_button:
                    action = "TOGGLE"
                else:
                    action = "MOVE_TO_BUTTON"

        if isinstance(action, str):
            action = self.action_dict[action]
        # print(self.state, self.action_names[action])

        return action

    def act_randomly(self):
        return np.random.randint(self.action_dim)
