import os
import time
import random
import numpy as np
from copy import deepcopy

from env.kitchen import Kitchen
from env.kitchen_skills import DropSkill, MoveSkill, GraspSkill, ToggleSkill


class KitchenWithSkill(Kitchen):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        mount_types="default",
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        table_full_size=(1.0, 0.8, 0.05),
        table_offset=(-0.2, 0, 0.90),
        butter_x_range=(0.2, 0.3),
        butter_y_range=(-0.3, 0.0),
        meatball_x_range=(0.2, 0.3),
        meatball_y_range=(-0.3, 0.0),
        pot_x_range=(0.07, 0.07),
        pot_y_range=(-0.05, -0.05),
        button_x_range=(0.07, 0.07),
        button_y_range=(-0.05, -0.05),
        stove_x_range=(0.07, 0.07),
        stove_y_range=(-0.05, -0.05),
        target_x_range=(0.07, 0.07),
        target_y_range=(-0.05, -0.05),
        normalization_range=((-0.5, -0.5, 0.7), (0.5, 0.5, 1.1))
    ):

        self.skill_timestep = 0
        self.pre_grasp_dist_thre = 0.025
        self.pre_toggle_dist_thre = 0.05
        self.move_to_drop_pot_z_offset = 0.05
        self.move_to_drop_in_pot_z_offset = 0.10
        self.move_to_toggle_z_offset = 0.1
        self.slice_dict = None

        self.has_renderer = has_renderer
        self.action_names = ["GRASP_BUTTER", "GRASP_MEATBALL", "GRASP_POT_HANDLE",
                             "MOVE_TO_BUTTER", "MOVE_TO_MEATBALL", "MOVE_TO_POT", "MOVE_TO_POT_HANDLE",
                             "MOVE_TO_STOVE", "MOVE_TO_BUTTON", "MOVE_TO_TARGET", # "MOVE_TO_RANDOM_PLACE",
                             "DROP", "TOGGLE", "NOOP"]
        self.num_skills = len(self.action_names)

        self.prev_skill_action = np.zeros(4)

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            use_object_obs=use_object_obs,
            reward_scale=reward_scale,
            placement_initializer=placement_initializer,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=True,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            table_full_size=table_full_size,
            table_offset=table_offset,
            butter_x_range=butter_x_range,
            butter_y_range=butter_y_range,
            meatball_x_range=meatball_x_range,
            meatball_y_range=meatball_y_range,
            pot_x_range=pot_x_range,
            pot_y_range=pot_y_range,
            button_x_range=button_x_range,
            button_y_range=button_y_range,
            stove_x_range=stove_x_range,
            stove_y_range=stove_y_range,
            target_x_range=target_x_range,
            target_y_range=target_y_range,
            normalization_range=normalization_range
        )

    def adjust_step_count(self, obs):
        obs["step_count"] = np.array([float(self.skill_timestep) / self.horizon])
        return obs

    def normalize_obs(self, obs, out_of_range_warning=True):
        self.prev_unnormalized_obs = deepcopy(obs)
        return super().normalize_obs(obs, out_of_range_warning)

    def reset(self):
        self.skill_timestep = 0
        obs = super().reset()
        return self.adjust_step_count(obs)

    def get_grasp_status(self, state):
        if state["butter_grasped"]:
            thing_in_hand = True
            in_hand_obj_pos_key = "butter_pos"
            in_hand_obj_touch_key = "butter_grasped"
        elif state["meatball_grasped"]:
            thing_in_hand = True
            in_hand_obj_pos_key = "meatball_pos"
            in_hand_obj_touch_key = "meatball_grasped"
        elif state["pot_grasped"]:
            thing_in_hand = True
            in_hand_obj_pos_key = "pot_pos"
            in_hand_obj_touch_key = "pot_touched"
        else:
            thing_in_hand = False
            in_hand_obj_pos_key = "robot0_eef_pos"
            in_hand_obj_touch_key = None
        return thing_in_hand, in_hand_obj_pos_key, in_hand_obj_touch_key

    def step(self, action):
        state = self.prev_unnormalized_obs

        # update cooking status
        if self.button_on and self.pot_on_stove:
            if self.butter_in_pot:
                prev_butter_melt_status = self.butter_melt_status
                self.butter_melt_status = min(self.butter_melt_status + 0.2, 1)
                if prev_butter_melt_status < 1 and self.butter_melt_status == 1:
                    butter_obj = self.objects_dict["butter"]
                    body_id = self.obj_body_id["butter"]
                    self.sim.data.set_joint_qpos(butter_obj.joints[0], np.concatenate([np.array((0, 0, 0)), self.sim.model.body_quat[body_id]]))
            if self.meatball_in_pot:
                if self.butter_melt_status != 1:
                    self.meatball_overcooked = True
                elif not self.meatball_overcooked:
                    self.meatball_cook_status = min(self.meatball_cook_status + 0.2, 1)

        do_nothing = False
        thing_in_hand, in_hand_obj_pos_key, in_hand_obj_touch_key = self.get_grasp_status(state)

        eef_pos = state["robot0_eef_pos"]

        skill = None
        action_name = self.action_names[action]
        if "GRASP" in action_name:
            if action_name == "GRASP_BUTTER":
                obj_pos_key, obj_grasped_key = "butter_pos", "butter_grasped"
            elif action_name == "GRASP_MEATBALL":
                obj_pos_key, obj_grasped_key = "meatball_pos", "meatball_grasped"
            elif action_name == "GRASP_POT_HANDLE":
                obj_pos_key, obj_grasped_key = "pot_handle_pos", "pot_grasped"
            else:
                raise NotImplementedError

            eef_obj_dist = np.linalg.norm(eef_pos - state[obj_pos_key])
            do_nothing = thing_in_hand or eef_obj_dist > self.pre_grasp_dist_thre

            skill = GraspSkill(obj_pos_key, obj_grasped_key)
        elif "MOVE_TO" in action_name:
            z_offset = 0
            move_to_grasp = False
            if action_name == "MOVE_TO_BUTTER":
                target_pos_key = "butter_pos"
                move_to_grasp = True
            elif action_name == "MOVE_TO_MEATBALL":
                target_pos_key = "meatball_pos"
                move_to_grasp = True
            elif action_name == "MOVE_TO_POT":
                target_pos_key = "pot_pos"
                z_offset = self.move_to_drop_in_pot_z_offset
            elif action_name == "MOVE_TO_POT_HANDLE":
                target_pos_key = "pot_handle_pos"
                move_to_grasp = True
            elif action_name == "MOVE_TO_STOVE":
                target_pos_key = "stove_pos"
                z_offset = self.move_to_drop_pot_z_offset
            elif action_name == "MOVE_TO_BUTTON":
                target_pos_key = "button_pos"
                z_offset = self.move_to_toggle_z_offset
            elif action_name == "MOVE_TO_TARGET":
                target_pos_key = "target_pos"
                z_offset = self.move_to_drop_pot_z_offset
            # elif action_name == "MOVE_TO_RANDOM_PLACE":
            #     low = self.global_low + 0.2
            #     low[2] = min(self.table_offset[2] + 0.1, low[2])
            #     high = self.global_high - 0.2
            #     target_pos_key = np.random.uniform(low, high)
            else:
                raise NotImplementedError

            if move_to_grasp or action_name == "MOVE_TO_BUTTON":
                do_nothing = thing_in_hand
                if action_name == "MOVE_TO_BUTTER" and self.butter_melt_status == 1:
                    do_nothing = True
            else:
                do_nothing = isinstance(target_pos_key, str) and in_hand_obj_pos_key == target_pos_key

            skill = MoveSkill(in_hand_obj_pos_key, target_pos_key, z_offset, thing_in_hand)
        elif action_name == "DROP":
            skill = DropSkill(in_hand_obj_touch_key)
        elif action_name == "TOGGLE":
            button_pos = state["button_pos"]
            eef_button_xy_dist = np.linalg.norm(eef_pos[:2] - button_pos[:2])
            do_nothing = thing_in_hand or eef_button_xy_dist > self.pre_toggle_dist_thre
            skill = ToggleSkill(not self.button_on)
        elif action_name == "NOOP":
            do_nothing = True
        else:
            raise NotImplementedError

        if do_nothing:
            action = np.zeros(4)
            action[-1] = self.prev_skill_action[-1]
            next_obs, reward, _, info = super().step(action)
            if self.has_renderer:
                self.render()
                time.sleep(0.01)
        else:
            skill_done = False
            while not skill_done:
                skill_action, skill_done = skill.step(self.prev_unnormalized_obs)
                next_obs, reward, _, info = super().step(skill_action)
                self.prev_skill_action = skill_action
                if self.has_renderer:
                    self.render()
                    time.sleep(0.01)

        self.skill_timestep += 1
        done = self.skill_timestep >= self.horizon

        evaluate_mask = True
        if evaluate_mask:
            info["local_causality"] = self.evaluate_mask(skill, do_nothing, state, self.prev_unnormalized_obs)

        return self.adjust_step_count(next_obs), reward, done, info

    def fill_mask(self, mask, childrens, parents):
        for child in childrens:
            if isinstance(child, str):
                child = self.slice_dict[child]
            for parent in parents:
                mask[child, self.slice_dict[parent]] = True

    def evaluate_mask(self, skill, do_nothing, state, next_state):
        if self.slice_dict is None:
            dynamics_keys = ["robot0_eef_pos", "robot0_gripper_qpos",
                             "butter_pos", "butter_quat", "butter_melt_status",
                             "meatball_pos", "meatball_cook_status", "meatball_overcooked",
                             "pot_pos", "pot_quat",
                             "stove_pos", "target_pos",
                             "button_pos", "button_joint_qpos"]

            slice_dict, cum = {}, 0
            for k in dynamics_keys:
                v = state[k]
                k_dim = 1 if np.isscalar(v) or v.ndim == 0 else len(v)
                slice_dict[k] = slice(cum, cum + k_dim)
                cum += k_dim
            slice_dict["action"] = slice(cum, cum + 1)
            self.slice_dict, self.feature_dim = slice_dict, cum

        mask = np.zeros((self.feature_dim, self.feature_dim + 1), dtype=bool)

        # stove dependencies
        if state["butter_melt_status"] != next_state["butter_melt_status"]:
            childrens, parents = ["butter_melt_status"], ["butter_pos", "pot_pos", "stove_pos", "button_joint_qpos", "butter_melt_status"]
            if state["butter_melt_status"] < 1 and next_state["butter_melt_status"] == 1:
                childrens.extend(["butter_pos", "butter_quat"])
            self.fill_mask(mask, childrens, parents)
        if state["meatball_overcooked"] != next_state["meatball_overcooked"]:
            childrens, parents = ["meatball_overcooked"], ["meatball_pos", "pot_pos", "stove_pos", "button_joint_qpos", "butter_melt_status"]
            self.fill_mask(mask, childrens, parents)
        if state["meatball_cook_status"] != next_state["meatball_cook_status"]:
            childrens, parents = ["meatball_cook_status"], ["meatball_pos", "pot_pos", "stove_pos", "button_joint_qpos", "butter_melt_status", "meatball_overcooked"]
            self.fill_mask(mask, childrens, parents)

        if do_nothing:
            return mask

        thing_in_hand, in_hand_obj_pos_key, in_hand_obj_touch_key = self.get_grasp_status(state)

        childrens, parents = [], []
        if isinstance(skill, GraspSkill):
            if next_state[skill.grasped_key]:
                childrens, parents = ["robot0_gripper_qpos"], ["robot0_eef_pos", "action"]

                obj_pos_key = skill.obj_pos_key
                if obj_pos_key == "pot_handle_pos":
                    parents.extend(["pot_pos", "pot_quat"])
                    childrens.append("pot_quat")
                else:
                    parents.append(obj_pos_key)
                    if obj_pos_key == "butter_pos":
                        childrens.append("butter_quat")

        elif isinstance(skill, MoveSkill):
            childrens, parents = ["robot0_eef_pos"], ["action"]

            target_pos_key = skill.target_pos_key
            if isinstance(target_pos_key, str):
                if target_pos_key == "pot_handle_pos":
                    parents.extend(["pot_pos", "pot_quat"])
                else:
                    parents.append(target_pos_key)

            if in_hand_obj_pos_key != "robot0_eef_pos":
                childrens.append(in_hand_obj_pos_key)
                if in_hand_obj_pos_key == "pot_pos":
                    if self.butter_in_pot:
                        childrens.append("butter_pos")
                    if self.meatball_in_pot:
                        childrens.append("meatball_pos")
        elif isinstance(skill, DropSkill):
            if (state["robot0_gripper_qpos"] != next_state["robot0_gripper_qpos"]).any():
                childrens, parents = ["robot0_gripper_qpos"], ["action"]
                if in_hand_obj_pos_key != "robot0_eef_pos":
                    childrens.append(self.slice_dict[in_hand_obj_pos_key].start + 2)
                    if in_hand_obj_pos_key == "pot_pos":
                        if self.butter_in_pot:
                            childrens.append(self.slice_dict["butter_pos"].start + 2)
                        if self.meatball_in_pot:
                            childrens.append(self.slice_dict["meatball_pos"].start + 2)
        elif isinstance(skill, ToggleSkill):
            childrens, parents = ["robot0_eef_pos", "button_joint_qpos"], ["button_pos", "button_joint_qpos", "action"]
        else:
            raise NotImplementedError

        self.fill_mask(mask, childrens, parents)

        return mask
