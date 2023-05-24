"""
modified from
https://github.com/ARISE-Initiative/robosuite/blob/maple/robosuite/controllers/skill_controller.py
and
https://github.com/ARISE-Initiative/robosuite/blob/maple/robosuite/controllers/skills.py
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.utils import to_numpy


class BaseSkill:
    def __init__(self, skill_params):
        global_xyz_bound = np.array(skill_params.global_xyz_bound)
        self.global_param_bound = global_xyz_bound.copy()
        self.global_low = global_xyz_bound[0]
        self.global_high = global_xyz_bound[1]
        self.reach_threshold = skill_params.reach_threshold
        self.lift_height = skill_params.lift_height

        self.disturbance_prob = skill_params.disturbance_prob

        self.num_required_block_steps = skill_params.num_block_steps
        self.prev_eef = None

        obj_name_mapper = {}
        env_params = skill_params.env_params
        self.env_name = env_name = env_params.env_name
        if "ToolUse" in env_name:
            obj_name_mapper = {0: "cube", 1: "tool", 2: "pot"}
            num_objs = 3
        elif env_name == "Kitchen":
            obj_name_mapper = {0: "cube", 1: "pot", 2: "button"}
            num_objs = 3
        elif "Block" in env_name:
            num_movable_objects = env_params.manipulation_env_params.block_env_params.num_movable_objects
            obj_name_mapper.update({i: "mov{}".format(i) for i in range(num_movable_objects)})
            num_objs = num_movable_objects
            self.num_movable_objects = num_movable_objects
        else:
            raise ValueError("Unknown env_name: {}".format(env_name))

        self.obj_name_mapper = obj_name_mapper
        self.is_object_oriented_skill = False
        self.is_object_necessary = False
        self.requires_grasp = False
        self.num_objs = num_objs

        self.controller_scale = skill_params.controller_scale

        self.init_obs = None
        self.obj = None
        self.params = None
        self.num_skill_params = 0
        self.num_max_steps = 0
        self.num_step = 0

        self.STATES = self.state = None

        self.POS_NOISE = 0.2
        self.EEF_BLOCK_THRESHOLD = 1e-3

    def update_state(self, obs):
        pass

    def update_state_if_blocked(self, obs):
        if not self.STATES:
            return

        assert self.state is not None
        eef = obs["robot0_eef_pos"]
        if self.prev_eef is None or np.linalg.norm(eef - self.prev_eef) > self.EEF_BLOCK_THRESHOLD or \
            self.state in ["REACHED", "GRASPING", "RELEASING"]:
            self.prev_eef = eef
            self.num_block_steps = 0
        else:
            self.num_block_steps += 1

        if self.num_block_steps >= self.num_required_block_steps:
            current_state_idx = self.STATES.index(self.state)
            if current_state_idx:
                self.state = self.STATES[current_state_idx - 1]

    def reset(self, obs, obj, params):
        self.init_obs = obs
        self.obj = obj
        self.params = params
        self.num_step = 0

        self.is_disturbed = np.random.random() < self.disturbance_prob
        self.disturbance = np.random.uniform(-1, 1, 3)
        self.disturbance[2] = np.abs(self.disturbance[2])

    def get_obj_pos(self, obs, obj_name=None):
        if obj_name is None:
            obj_name = self.obj_name_mapper.get(self.obj, None)
        if obj_name:
            if obj_name == "pot":
                if self.requires_grasp:
                    return obs["pot_handle_pos"].copy()
                else:
                    pot_half_height = 0.02          # hardcoded
                    return obs["pot_pos"] + np.array([0, 0, pot_half_height])
            else:
                return obs[obj_name + "_pos"].copy()
        else:
            return None

    def get_obj_grasped(self, obs, obj_name=None):
        if obj_name is None:
            obj_name = self.obj_name_mapper.get(self.obj, None)
        if obj_name:
            return obs[obj_name + "_grasped"].copy()
        else:
            return None

    def get_global_pos(self, scale):
        low, high = self.global_param_bound
        return (high + low) / 2 + scale * (high - low) / 2

    def get_delta_pos(self, eef, goal_pos):
        return np.clip((goal_pos - eef) / self.controller_scale, -1, 1)

    def get_pos_ac(self, obs):
        raise NotImplementedError

    def get_gripper_ac(self, obs):
        raise NotImplementedError

    def is_success(self):
        return False

    def step(self, obs):
        self.update_state(obs)
        self.update_state_if_blocked(obs)

        pos = self.get_pos_ac(obs)
        mask = np.abs(pos) >= 3 * self.POS_NOISE
        disturbed_pos = np.clip(pos, -1 + self.POS_NOISE, 1 - self.POS_NOISE) + np.random.uniform(-self.POS_NOISE, self.POS_NOISE, 3)
        pos = np.where(mask, disturbed_pos, pos)

        # eef after step shouldn't be out of global bound
        eef = obs["robot0_eef_pos"]
        pos = np.clip(pos, -1, 1)
        pos = np.clip(pos, (self.global_low - eef) / self.controller_scale, (self.global_high - eef) / self.controller_scale)

        grp = self.get_gripper_ac(obs)
        grp = grp * (1 + np.random.rand()) / 2
        self.num_step += 1
        return np.concatenate([pos, grp]), self.num_step >= self.num_max_steps or self.is_success()


class AtomicSkill(BaseSkill):
    def __init__(self, skill_params):
        super().__init__(skill_params)
        self.is_object_oriented_skill = False
        self.is_object_necessary = False
        self.num_skill_params = 4
        self.num_max_steps = 1

    def get_pos_ac(self, obs):
        return self.params[:3].copy()

    def get_gripper_ac(self, obs):
        return self.params[3:].copy()


class GripperSkill(BaseSkill):
    def __init__(self, skill_params, is_open):
        super().__init__(skill_params)
        self.is_open = is_open
        self.is_object_oriented_skill = False
        self.is_object_necessary = False
        self.num_skill_params = 0
        self.num_max_steps = skill_params.gripper_skill_params.num_max_steps

    def get_pos_ac(self, obs):
        return np.zeros(3)

    def get_gripper_ac(self, info):
        gripper_action = np.ones(1)
        if self.is_open:
            gripper_action *= -1
        return gripper_action


class OpenSkill(GripperSkill):
    def __init__(self, skill_params):
        super().__init__(skill_params, is_open=True)


class CloseSkill(GripperSkill):
    def __init__(self, skill_params):
        super().__init__(skill_params, is_open=False)


class LiftSkill(BaseSkill):

    def __init__(self, skill_params):
        super().__init__(skill_params)
        lift_skill_params = skill_params.lift_skill_params
        if hasattr(lift_skill_params, "global_param_bound"):
            self.global_param_bound = np.array(lift_skill_params.global_param_bound)

        self.is_object_oriented_skill = True
        self.is_object_necessary = True
        self.requires_grasp = True
        self.num_skill_params = 3
        self.num_max_steps = lift_skill_params.num_max_steps
        self.num_required_reach_steps = lift_skill_params.num_reach_steps
        self.num_required_grasp_steps = lift_skill_params.num_grasp_steps

        self.state = "LIFTING"
        self.num_reach_steps = 0
        self.num_grasp_steps = 0
        self.num_arrive_steps = 0

        self.STATES = ["LIFTING", "HOVERING", "LOWERING", "REACHED", "GRASPING", "ARRIVING"]

    def reset(self, *args, **kwargs):
        super().reset(*args, *kwargs)
        self.state = "LIFTING"
        self.num_reach_steps = 0
        self.num_grasp_steps = 0
        self.num_arrive_steps = 0
        if self.obj >= self.num_objs:
            self.obj = 0
        if "Block" in self.env_name:
            self.obj = np.random.randint(self.num_movable_objects)

        self.disturbance[2] *= np.random.random() > 0.5

    def get_disturbance(self):
        if self.is_disturbed:
            disturbance = self.disturbance.copy()
            disturbance_min = np.array([0.05, 0.05, 0.03])
            disturbance_max = np.array([0.08, 0.08, 0.06])
            return (disturbance_max - disturbance_min) * self.disturbance + disturbance_min * np.sign(self.disturbance)
        else:
            return 0

    def update_state(self, obs):
        eef = obs["robot0_eef_pos"]
        obj_pos = self.get_obj_pos(obs)
        obj_grasped = self.get_obj_grasped(obs)
        goal_pos = self.get_global_pos(self.params)
        assert obj_pos is not None

        obj_pos += self.get_disturbance()
        if self.is_disturbed:
            obj_grasped = self.num_grasp_steps >= self.num_required_grasp_steps

        th = self.reach_threshold
        reached_lift = (eef[2] >= self.lift_height - th)
        reached_xy = (np.linalg.norm(eef[0:2] - obj_pos[0:2]) < th)
        reached_xyz = (np.linalg.norm(eef - obj_pos) < th)
        reached_goal = (np.linalg.norm(eef - goal_pos) < th)

        if self.state == "ARRIVING" or (self.state == "GRASPING" and obj_grasped):
            self.state = "ARRIVING"
            self.num_arrive_steps += reached_goal
        elif self.state == "GRASPING" or \
                (self.state == "REACHED" and self.num_reach_steps >= self.num_required_reach_steps):
            self.state = "GRASPING"
            self.num_grasp_steps += 1
        elif self.state == "REACHED" or (self.state == "LOWERING" and reached_xyz):
            self.state = "REACHED"
            self.num_reach_steps += 1
        elif self.state == "LOWERING" or (self.state == "HOVERING" and reached_xy):
            self.state = "LOWERING"
        elif self.state == "HOVERING" or (self.state == "LIFTING" and reached_lift):
            self.state = "HOVERING"
        else:
            self.state = "LIFTING"

        assert self.state in self.STATES

    def get_pos_ac(self, obs):
        eef = obs["robot0_eef_pos"]
        if self.state == "ARRIVING":
            goal_pos = self.get_global_pos(self.params)
        else:
            goal_pos = self.get_obj_pos(obs)
            assert goal_pos is not None

            goal_pos += self.get_disturbance()

        if self.state == "LIFTING":
            goal_pos[2] = self.lift_height + self.controller_scale
        elif self.state == "HOVERING":
            goal_pos[2] = max(self.lift_height, eef[-1])

        pos = self.get_delta_pos(eef, goal_pos)

        return pos

    def get_gripper_ac(self, info):
        if self.state in ["GRASPING", "ARRIVING"]:
            gripper_action = np.ones(1)
        else:
            gripper_action = np.ones(1) * -1
        return gripper_action

    def is_success(self):
        return self.num_arrive_steps >= 2


class PickPlaceSkill(BaseSkill):

    def __init__(self, skill_params):
        super().__init__(skill_params)
        pick_place_skill_params = skill_params.pick_place_skill_params
        if hasattr(pick_place_skill_params, "global_param_bound"):
            self.global_param_bound = np.array(pick_place_skill_params.global_param_bound)

        self.is_object_oriented_skill = True
        self.is_object_necessary = True
        self.num_skill_params = 0

        self.place_target_name = pick_place_skill_params.place_target_name
        self.num_max_steps = pick_place_skill_params.num_max_steps
        self.num_required_reach_steps = pick_place_skill_params.num_reach_steps
        self.num_required_grasp_steps = pick_place_skill_params.num_grasp_steps
        self.num_required_arrived_steps = pick_place_skill_params.num_arrived_steps
        self.num_required_release_steps = pick_place_skill_params.num_release_steps

        self.state = "LIFTING"
        self.num_reach_steps = 0
        self.num_grasp_steps = 0
        self.num_arrive_steps = 0
        self.num_release_steps = 0

        self.STATES = ["LIFTING", "HOVERING", "LOWERING", "REACHED", "GRASPING",
                       "PICK_LIFTING", "ARRIVING", "RELEASING"]

    def reset(self, *args, **kwargs):
        super().reset(*args, *kwargs)
        self.state = "LIFTING"
        self.num_reach_steps = 0
        self.num_grasp_steps = 0
        self.num_arrive_steps = 0
        self.num_release_steps = 0
        self.obj = 0    # only pick the cube

    def get_disturbance(self):
        if self.is_disturbed:
            disturbance_min = 0.0
            disturbance_max = 0.1
            return (disturbance_max - disturbance_min) * self.disturbance + disturbance_min * np.sign(self.disturbance)
        else:
            return 0

    def update_state(self, obs):
        eef = obs["robot0_eef_pos"]
        obj_pos = self.get_obj_pos(obs)
        obj_grasped = self.get_obj_grasped(obs)
        target_pos = self.get_obj_pos(obs, obj_name=self.place_target_name)

        target_pos += self.get_disturbance()

        th = self.reach_threshold
        reached_lift = (eef[2] >= self.lift_height - th)
        reached_xy = (np.linalg.norm(eef[0:2] - obj_pos[0:2]) < th)
        reached_xyz = (np.linalg.norm(eef - obj_pos) < th)
        reached_target_xy = (np.linalg.norm(eef[0:2] - target_pos[0:2]) < th)

        if self.state == "RELEASING" or \
                (self.state == "ARRIVING" and self.num_arrive_steps >= self.num_required_arrived_steps):
            self.state = "RELEASING"
            self.num_release_steps += (np.linalg.norm(eef - obj_pos) > 0.05)
        elif self.state == "ARRIVING" or \
                (self.state == "PICK_LIFTING" and reached_lift):
            self.state = "ARRIVING"
            self.num_arrive_steps += reached_target_xy
        elif self.state == "PICK_LIFTING" or \
                (self.state == "GRASPING" and obj_grasped):
            self.state = "PICK_LIFTING"
        elif self.state == "GRASPING" or \
                (self.state == "REACHED" and (self.num_reach_steps >= self.num_required_reach_steps)):
            self.state = "GRASPING"
            self.num_grasp_steps += 1
        elif self.state == "REACHED" or (self.state == "LOWERING" and reached_xyz):
            self.state = "REACHED"
            self.num_reach_steps += 1
        elif self.state == "LOWERING" or (self.state == "HOVERING" and reached_xy):
            self.state = "LOWERING"
        elif self.state == "HOVERING" or (self.state == "LIFTING" and reached_lift):
            self.state = "HOVERING"
        else:
            self.state = "LIFTING"

        assert self.state in self.STATES

    def get_pos_ac(self, obs):
        eef = obs["robot0_eef_pos"]
        if self.state == "ARRIVING":
            goal_pos = self.get_obj_pos(obs, obj_name=self.place_target_name)
            goal_pos += self.get_disturbance()
        else:
            goal_pos = self.get_obj_pos(obs)
            assert goal_pos is not None

        if self.state in ["LIFTING", "PICK_LIFTING"]:
            goal_pos[2] = self.lift_height + self.controller_scale
        elif self.state in ["HOVERING", "ARRIVING"]:
            goal_pos[2] = max(self.lift_height, eef[-1])

        pos = self.get_delta_pos(eef, goal_pos)

        return pos

    def get_gripper_ac(self, info):
        if self.state in ["GRASPING", "PICK_LIFTING", "ARRIVING"]:
            gripper_action = np.ones(1)
        else:
            gripper_action = np.ones(1) * -1
        return gripper_action

    def is_success(self):
        return self.num_release_steps >= self.num_required_release_steps


class HookSkill(BaseSkill):

    def __init__(self, skill_params):
        super().__init__(skill_params)
        hook_skill_params = skill_params.hook_skill_params
        if hasattr(hook_skill_params, "global_param_bound"):
            self.global_param_bound = np.array(hook_skill_params.global_param_bound)

        self.is_object_oriented_skill = True
        self.is_object_necessary = True
        self.num_skill_params = 3

        self.delta_xyz_scale = np.array(hook_skill_params.delta_xyz_scale)
        self.tool_name = hook_skill_params.tool_name
        self.tool_relative_pos = hook_skill_params.tool_relative_pos
        self.num_max_steps = hook_skill_params.num_max_steps
        self.num_required_reach_steps = hook_skill_params.num_reach_steps
        self.num_required_grasp_steps = hook_skill_params.num_grasp_steps

        self.state = "LIFTING"
        self.num_reach_steps = 0
        self.num_grasp_steps = 0
        self.num_arrive_steps = 0

        self.STATES = ["LIFTING", "HOVERING", "LOWERING", "REACHED", "GRASPING",
                       "LOCATING", "MOVING", "PUSHING", "RETURNING"]

    def reset(self, *args, **kwargs):
        super().reset(*args, *kwargs)
        self.params[0] = -np.abs(self.params[0])
        self.params[1] = 0
        self.state = "LIFTING"
        self.num_grasp_steps = 0
        self.num_reach_steps = 0
        self.num_arrive_steps = 0
        self.obj = 0

    def update_state(self, obs):
        eef = obs["robot0_eef_pos"]
        obj_pos = self.get_obj_pos(obs)
        tool_pos = self.get_obj_pos(obs, obj_name=self.tool_name)
        tool_grasped = self.get_obj_grasped(obs, obj_name=self.tool_name)
        tool_head_pos = obs["tool_head_pos"]
        tool_head_goal_pos = obj_pos + np.array([0.1, -0.1, 0])
        start, target = self.get_start_target()

        if self.is_disturbed:
            disturbance_min = 0.05
            disturbance_max = 0.1
            tool_pos[:2] += (disturbance_max - disturbance_min) * self.disturbance[:2] + \
                            disturbance_min * np.sign(self.disturbance[:2])
            tool_grasped = self.num_grasp_steps >= self.num_required_grasp_steps
            tool_head_pos = eef + tool_head_pos - tool_pos

        th = self.reach_threshold
        reached_lift = eef[2] >= (self.lift_height - th)
        reached_xy = np.linalg.norm(eef[0:2] - tool_pos[0:2]) < th
        reached_xyz = np.linalg.norm(eef - tool_pos) < th
        reached_locating_goal = np.linalg.norm(tool_head_pos - tool_head_goal_pos) < th
        reached_start = np.linalg.norm(tool_head_pos - start) < th
        reached_target = np.linalg.norm(tool_head_pos - target) < th

        if self.state == "RETURNING" or (self.state == "PUSHING" and reached_target):
            self.state = "RETURNING"
            self.num_arrive_steps += reached_start
        elif self.state == "PUSHING" or (self.state == "MOVING" and reached_start):
            self.state = "PUSHING"
        elif self.state == "MOVING" or (self.state == "LOCATING" and reached_locating_goal):
            self.state = "MOVING"
        elif self.state == "LOCATING" or \
                (self.state == "GRASPING" and tool_grasped):
            self.state = "LOCATING"
        elif self.state == "GRASPING" or \
                (self.state == "REACHED" and (self.num_reach_steps >= self.num_required_reach_steps)):
            self.state = "GRASPING"
            self.num_grasp_steps += 1
        elif self.state == "REACHED" or (self.state == "LOWERING" and reached_xyz):
            self.state = "REACHED"
            self.num_reach_steps += 1
        elif self.state == "LOWERING" or (self.state == "HOVERING" and reached_xy):
            self.state = "LOWERING"
        elif self.state == "HOVERING" or (self.state == "LIFTING" and reached_lift):
            self.state = "HOVERING"
        else:
            self.state = "LIFTING"

        assert self.state in self.STATES

    def get_start_target(self,):
        goal_pos = self.get_obj_pos(self.init_obs)
        assert goal_pos is not None
        start = np.clip(goal_pos - self.params * self.delta_xyz_scale, self.global_low, self.global_high)
        target = np.clip(goal_pos + self.params * self.delta_xyz_scale, self.global_low, self.global_high)
        return start, target

    def get_pos_ac(self, obs):
        eef = obs["robot0_eef_pos"]
        obj_pos = self.get_obj_pos(obs)
        tool_pos = self.get_obj_pos(obs, obj_name=self.tool_name)
        tool_head_pos = obs["tool_head_pos"]
        tool_head_goal_pos = obj_pos + np.array([0.1, -0.1, 0])
        start, target = self.get_start_target()

        if self.is_disturbed:
            disturbance_min = 0.05
            disturbance_max = 0.1
            tool_pos[:2] += (disturbance_max - disturbance_min) * self.disturbance[:2] + \
                            disturbance_min * np.sign(self.disturbance[:2])
            tool_head_pos = eef + tool_head_pos - tool_pos

        if self.state in ["LIFTING", "HOVERING", "LOWERING", "REACHED", "GRASPING"]:
            move_pos, goal_pos = eef, tool_pos
        elif self.state in ["LOCATING"]:
            move_pos, goal_pos = tool_head_pos, tool_head_goal_pos
        elif self.state in ["PUSHING"]:
            move_pos, goal_pos = tool_head_pos, target
        elif self.state in ["MOVING", "RETURNING"]:
            move_pos, goal_pos = tool_head_pos, start
        else:
            raise NotImplementedError

        if self.state == "LIFTING":
            goal_pos[2] = self.lift_height + self.controller_scale
        elif self.state == "HOVERING":
            goal_pos[2] = max(self.lift_height, eef[-1])

        pos = self.get_delta_pos(move_pos, goal_pos)

        return pos

    def get_gripper_ac(self, info):
        if self.state in ["GRASPING", "LOCATING", "MOVING", "PUSHING", "RETURNING"]:
            gripper_action = np.ones(1)
        else:
            gripper_action = np.ones(1) * -1
        return gripper_action

    def is_success(self):
        return self.num_arrive_steps


class PushSkill(BaseSkill):

    def __init__(self, skill_params):
        super().__init__(skill_params)
        push_skill_params = skill_params.push_skill_params
        if hasattr(push_skill_params, "global_param_bound"):
            self.global_param_bound = np.array(push_skill_params.global_param_bound)

        self.is_object_oriented_skill = True
        self.is_object_necessary = True
        self.num_skill_params = 3
        self.num_max_steps = push_skill_params.num_max_steps
        self.delta_xyz_scale = np.array(push_skill_params.delta_xyz_scale)

        self.state = "LIFTING"
        self.num_reach_steps = 0

        self.STATES = ["LIFTING", "HOVERING", "LOWERING", "PUSHING"]

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.state = "LIFTING"
        if self.obj >= self.num_objs:
            self.obj = 0
        if "ToolUse" in self.env_name:
            self.obj = 0
        self.num_reach_steps = 0

        if np.abs(self.params[0]) > np.abs(self.params[1]):
            # if mainly move along x axis, then disturb y
            self.disturbance[0] = 0
        else:
            self.disturbance[1] = 0
        self.disturbance[2] *= np.random.random() < 1 / 3

    def get_disturbance(self):
        if self.is_disturbed:
            xy_disturbance_max = 0.1
            disturbance_max = np.array([xy_disturbance_max, xy_disturbance_max, 0.08])
            return disturbance_max * self.disturbance
        else:
            return 0

    def update_state(self, obs):
        eef = obs["robot0_eef_pos"]
        start, target = self.get_start_target()

        th = self.reach_threshold
        reached_lift = (eef[2] >= self.lift_height - th)
        reached_src_xy = (np.linalg.norm(eef[0:2] - start[0:2]) < th)
        reached_src_xyz = (np.linalg.norm(eef - start) < th)
        reached_target_xyz = (np.linalg.norm(eef - target) < 0.02)

        if self.state == "PUSHING" or (self.state == "LOWERING" and reached_src_xyz):
            self.state = "PUSHING"
            self.num_reach_steps += reached_target_xyz
        elif self.state == "LOWERING" or (self.state == "HOVERING" and reached_src_xy):
            self.state = "LOWERING"
        elif self.state == "HOVERING" or (self.state == "LIFTING" and reached_lift):
            self.state = "HOVERING"
        else:
            self.state = "LIFTING"

        assert self.state in self.STATES

    def get_start_target(self):
        goal_pos = self.get_obj_pos(self.init_obs)
        assert goal_pos is not None
        start = np.clip(goal_pos - self.params * self.delta_xyz_scale, self.global_low, self.global_high)
        target = np.clip(goal_pos + self.params * self.delta_xyz_scale, self.global_low, self.global_high)

        disturbance = self.get_disturbance()
        start += disturbance
        target += disturbance

        return start, target

    def get_pos_ac(self, obs):
        eef = obs["robot0_eef_pos"]
        start, target = self.get_start_target()

        if self.state == "LIFTING":
            goal_pos = eef.copy()
            goal_pos[2] = self.lift_height + self.controller_scale
        elif self.state == "HOVERING":
            goal_pos = start.copy()
            goal_pos[2] = max(self.lift_height, eef[-1])
        elif self.state == "LOWERING":
            goal_pos = start.copy()
        elif self.state == "PUSHING":
            goal_pos = target.copy()
        else:
            raise NotImplementedError

        pos = self.get_delta_pos(eef, goal_pos)

        return pos

    def get_gripper_ac(self, obs):
        if self.state in ["LOWERING", "PUSHING"]:
            gripper_action = np.ones(1)
        else:
            gripper_action = np.ones(1) * -1
        return gripper_action

    def is_success(self):
        return self.num_reach_steps >= 2


class SkillController:

    SKILL_DICT = {"atomic": AtomicSkill,
                  "lift": LiftSkill,
                  "push": PushSkill,
                  "open": OpenSkill,
                  "close": CloseSkill,
                  "pick_place": PickPlaceSkill,
                  "hook": HookSkill}

    def __init__(self, params):
        self.params = params
        self.hippo_params = hippo_params = params.policy_params.hippo_params
        self.skill_params = skill_params = hippo_params.skill_params

        skill_params.env_params = params.env_params

        self.num_envs = params.env_params.num_envs
        self.skill_set = [[self.SKILL_DICT[skill_name](skill_params)
                           for skill_name in hippo_params.skill_names]
                          for _ in range(self.num_envs)]
        self.cur_skill = [None for _ in range(self.num_envs)]

    def update_config(self, i, obs, skill, obj, skill_param):
        self.cur_skill[i] = self.skill_set[i][skill]
        self.cur_skill[i].reset(obs, obj, skill_param)

    def get_action(self, obs):
        actions, dones = [], []
        for i in range(self.num_envs):
            obs_i = {k: v[i] for k, v in obs.items()}
            action, done = self.cur_skill[i].step(obs_i)
            actions.append(action)
            dones.append(done)
        return np.array(actions), np.array(dones)
