from env.cookbook import Cookbook

import os
import copy
import random
import numpy as np

from collections import OrderedDict


USE = 0
LEFT, RIGHT, UP, DOWN = [0, 1, 2, 3]
opposite_dir = {LEFT: RIGHT, RIGHT: LEFT, UP: DOWN, DOWN: UP}


class CraftWorld(object):
    def __init__(self, params):
        self.params = params
        self.env_params = env_params = params.env_params
        self.use_stage_reward = env_params.use_stage_reward
        self.craft_env_params = craft_env_params = env_params.craft_env_params

        self.goal = craft_env_params.goal
        self.width = craft_env_params.width
        self.height = craft_env_params.height
        self.horizon = craft_env_params.horizon
        self.use_pose = craft_env_params.use_pose

        recipe_path = os.path.join((os.path.dirname(os.path.realpath(__file__))), "craft_recipes", self.goal + "_recipe.yaml")
        self.cookbook = cookbook = Cookbook(recipe_path)

        self.environment_idxes = cookbook.environment_idxes
        self.primitive_idxes = cookbook.primitive_idxes
        self.craft_idxes = cookbook.craft_idxes

        self.environments = cookbook.environments
        self.primitives = cookbook.primitives
        self.recipes = cookbook.recipes

        self.has_furnace = "furnace" in self.cookbook.index
        self.goal_idx = cookbook.index[self.goal]
        self.boundary_idx = cookbook.index["boundary"]
        self.workshop_idx = cookbook.index["workshop"]
        self.furnace_idx = cookbook.index["furnace"]

        # for computing staged reward
        assert self.goal_idx in self.primitive_idxes
        goal_info = self.primitives[self.goal_idx]

        self.path_tool = None
        self.pick_tool = goal_info.get("_require", None)

        if "_surround" in goal_info:
            surround_idx = goal_info["_surround"]
            if surround_idx in self.environments:
                surround_info = self.environments[surround_idx]
            elif surround_idx in self.primitives:
                surround_info = self.primitives[surround_idx]
            else:
                raise NotImplementedError
            self.path_tool = surround_info.get("_require", None)

        self.craft_tools = {}
        self.inter_tools = {}
        self.num_craft_tool_stages = 0

        for tool_idx in [self.path_tool, self.pick_tool]:
            if tool_idx is None:
                continue
            recipe, num_stages = self.add_recipe(tool_idx)
            self.craft_tools[tool_idx] = [recipe, num_stages]
            self.num_craft_tool_stages += num_stages

        # Intialize actions: USE; move to treasure; move to workshop; move to walls; move to ingredients; craft tools 
        self.action_info_ready = False
        self.slice_dict = None
        self.dynamics_keys = None
        self.reset()
        self.obs_dims = self.observation_dims()


    def add_inter_tool(self, index):
        if index in self.inter_tools:
            return
        sub_recipe, num_sub_stages = self.add_recipe(index)
        self.inter_tools[index] = [sub_recipe, num_sub_stages]
        self.num_craft_tool_stages += num_sub_stages
        if index == self.furnace_idx:
            self.num_craft_tool_stages += 2    # 2 extra stages to set up the furnace

    def add_recipe(self, index):
        recipe_new, num_stages = {}, 0
        for k, v in self.recipes[index].items():
            if k in ["_at", "_yield", "_step"]:
                if k == "_at" and v == self.furnace_idx:
                    self.add_inter_tool(self.furnace_idx)
                continue
            elif k in self.primitive_idxes:
                recipe_new[k] = v
                num_stages += v
                tool_idx = self.primitives[k].get("_require", None)
                if tool_idx:
                    self.add_inter_tool(tool_idx)
            else:
                assert k in self.recipes, "Unknow key {}".format(k)
                sub_recipe, num_sub_stages = self.add_recipe(k)
                recipe_new[k] = [v, sub_recipe, num_sub_stages]
                num_stages += v * num_sub_stages
        return recipe_new, num_stages + 1   # 1 extra stage to craft the tool

    def tool_craftable(self, recipe):
        craftable = True
        for k, v in self.recipes[index].items():
            if k in ["_at", "_yield", "_step"]:
                if k == "_at" and v == self.furnace_idx and v not in self.inter_tools:
                    sub_recipe, num_sub_stages = self.add_recipe(self.furnace_idx)
                    self.inter_tools[v] = sub_recipe
                    self.num_craft_tool_stages += num_sub_stages + 2    # 2 extra stages to set up the furnace
                continue
            elif k in self.primitive_idxes:
                recipe_new[k] = v
                tool_idx = self.primitives[k].get("_require", None)
                if tool_idx and tool_idx not in self.inter_tools:
                    sub_recipe, num_sub_stages = self.add_recipe(tool_idx)
                    self.inter_tools[v] = sub_recipe
                    self.num_craft_tool_stages += num_sub_stages
            else:
                assert k in self.recipes, "Unknow key {}".format(k)
                sub_recipe, num_sub_stages = self.add_recipe(k)
                recipe_new[k] = [v, sub_recipe, num_sub_stages]
                num_stages += v * num_sub_stages
            num_stages += v
        return recipe_new, num_stages + 1   # 1 extra stage to craft the tool


    def reset(self):
        self.cur_step = self.stage_completion_tracker = 0

        assert self.goal_idx not in self.environments
        self.sample_scenario()

        return self.get_state()

    def neighbors(self, pos, dir=None):
        x, y = pos
        neighbors = []
        if x > 0 and (dir is None or dir == LEFT):
            neighbors.append((x - 1, y, LEFT))
        if y > 0 and (dir is None or dir == DOWN):
            neighbors.append((x, y - 1, DOWN))
        if x < self.width - 1 and (dir is None or dir == RIGHT):
            neighbors.append((x + 1, y, RIGHT))
        if y < self.height - 1 and (dir is None or dir == UP):
            neighbors.append((x, y + 1, UP))
        return neighbors

    def random_free(self, requires_free_neighbor=False):
        grid = self.grid
        pos = None
        while pos is None:
            x, y = np.random.randint(grid.shape[0]), np.random.randint(grid.shape[1])
            if grid[x, y, :].any():
                continue
            if requires_free_neighbor:
                if any([self.grid[nx, ny].any() for nx, ny, _ in self.neighbors((x, y))]):
                    continue
            pos = (x, y)
        return pos

    def sample_scenario(self):
        cookbook = self.cookbook

        # generate grid
        self.grid = grid = np.zeros((self.width, self.height, self.cookbook.n_kinds))
        i_bd = self.cookbook.index["boundary"]
        grid[0, :, i_bd] = 1
        grid[self.width - 1:, :, i_bd] = 1
        grid[:, 0, i_bd] = 1
        grid[:, self.height - 1:, i_bd] = 1

        self.state = OrderedDict()
        move_target_names = ["placeholder"]

        # treasure
        gx, gy = np.random.randint(1, self.width - 1), np.random.randint(1, 3)

        assert not grid[gx, gy].any()
        grid[gx, gy, self.goal_idx] = 1
        self.state[self.goal] = np.array([gx, gy])
        self.state[self.goal + "_faced"] = False
        self.state[self.goal + "_picked"] = False
        move_target_names.append(self.goal)

        obst_idx = self.primitives[self.goal_idx].get("_surround", None)
        if obst_idx:
            assert obst_idx in self.environments
            obst_name = cookbook.index[obst_idx]
            num_obsts = 0
            for ox, oy, _ in self.neighbors((gx, gy)):
                item_name = obst_name + str(num_obsts)
                self.state[item_name + "_faced"] = False
                if grid[ox, oy, :].any():
                    assert grid[ox, oy, :].argmax() == i_bd
                    self.state[item_name] = np.array([0, 0])
                    self.state[item_name + "_picked"] = True
                else:
                    grid[ox, oy, obst_idx] = 1
                    self.state[item_name] = np.array([ox, oy])
                    self.state[item_name + "_picked"] = False

                move_target_names.append(item_name)
                num_obsts += 1

        # ingredients
        for primitive_idx, primitive_info in self.primitives.items():
            if primitive_idx == self.goal_idx:
                continue
            for i in range(primitive_info["num"]):
                x, y = self.random_free(requires_free_neighbor=True)
                grid[x, y, primitive_idx] = 1
                primitive_name = cookbook.index[primitive_idx]
                item_name = primitive_name + str(i)
                self.state[item_name] = np.array([x, y])
                self.state[item_name + "_faced"] = False
                self.state[item_name + "_picked"] = False
                move_target_names.append(item_name)

        # generate crafting stations
        station_names = ["workshop"]

        has_furname = "furnace" in cookbook.index
        if has_furname:
            station_names.append("furnace")

        for station_name in station_names:
            x, y = self.random_free(requires_free_neighbor=True)
            grid[x, y, cookbook.index[station_name]] = 1
            self.state[station_name] = np.array([x, y])
            self.state[station_name + "_faced"] = False
            move_target_names.append(station_name)

        if has_furname:
            self.state["furnace_ready"] = False
            self.state["furnace_slot"] = 0
            self.state["furnace_stage"] = 0

        # generate init pos
        self.inventory = np.zeros(self.cookbook.n_kinds, dtype=int)
        self.pos = self.random_free()
        self.dir = np.random.randint(4)
        for x, y, _ in self.neighbors(self.pos, self.dir):
            if grid[x, y].any():
                item_name = self.find_key(x, y)
                if item_name in self.state:
                    self.state[item_name + "_faced"] = True

        # set up action information
        if not self.action_info_ready:
            self.action_info_ready = True
            self.move_target_names = move_target_names
            self.craft_action_starts = len(move_target_names)
            self.action_dim = self.craft_action_starts + len(self.craft_idxes)

            # for scripted policy
            self.move_actions = {name: i for i, name in enumerate(move_target_names) if name != "placeholder"}
            self.move_target_names = {i: name for i, name in enumerate(move_target_names) if name != "placeholder"}

    def get_state(self):
        state = copy.deepcopy(self.state)
        for k, v in state.items():
            if isinstance(v, (int, bool)):
                state[k] = np.array([v])

        if self.dynamics_keys is None:
            if self.use_pose:
                self.dynamics_keys = ["agent_pos", "agent_dir", "inventory"]
            else:
                self.dynamics_keys = ["agent_pos", "agent_dir"]

            for k in state:
                if k.endswith(("_faced", "_picked")):
                    if not self.use_pose:
                        self.dynamics_keys.append(k)
                else:
                    if self.use_pose and k not in self.dynamics_keys:
                        self.dynamics_keys.append(k)

            if not self.use_pose and self.has_furnace:
                self.dynamics_keys += ["furnace_ready", "furnace_slot", "furnace_stage"]

        state["agent_pos"] = np.array(self.pos)
        state["agent_dir"] = np.array([self.dir])
        state["inventory"] = self.inventory.copy()
        state["step_count"] = np.array([float(self.cur_step) / self.horizon])

        return state

    def observation_spec(self):
        return self.get_state()

    def observation_dims(self):
        state = {}
        for key in self.state:
            if key.endswith(("_faced", "_picked", "furnace_ready")):
                state[key] = np.array([2])
            elif key == "furnace_slot":
                state[key] = np.array([len(self.cookbook.idx2furnace_slot) + 1])
            elif key == "furnace_stage":
                state[key] = np.array([self.cookbook.furnace_max_stage + 1])
            else:
                state[key] = np.array([self.width, self.height])

        state["agent_pos"] = np.array([self.width, self.height])
        state["agent_dir"] = np.array([4])
        max_primitive_num = max([primitive_info["num"] for primitive_info in self.primitives.values()])
        state["inventory"] = np.ones_like(self.inventory, dtype=int) * (max_primitive_num + 1)
        state["step_count"] = np.array([1])

        return state

    def check_success(self):
        return self.inventory[self.goal_idx] > 0

    def has_craft(self, craft_idx):
        return self.inventory[craft_idx] > 0 or (craft_idx == self.furnace_idx and self.state["furnace_ready"])

    def can_collect_treasure(self):
        has_path = any([not self.grid[x, y].any() for x, y, _ in self.neighbors(self.state[self.goal])])
        if has_path:
            if self.pick_tool:
                return self.inventory[self.pick_tool] > 0
        else:
            if self.pick_tool and not self.inventory[self.pick_tool]:
                return False
            if self.path_tool and not self.inventory[self.path_tool]:
                return False
        return True

    def reward(self):
        num_required_tools = len(self.inter_tools) + len(self.craft_tools)
        if self.check_success():
            reward = 1
            self.stage_completion_tracker = num_required_tools + 2
        elif self.can_collect_treasure():
            reward = 0.75
            self.stage_completion_tracker = num_required_tools + 1
        else:
            self.stage_completion_tracker = 0
            for k in {**self.inter_tools, **self.craft_tools}:
                if self.has_craft(k):
                    self.stage_completion_tracker += 1
            reward = 0.5 * self.stage_completion_tracker / num_required_tools

        if self.use_stage_reward:
            return reward
        else:
            return float(reward == 1)

    def find_key(self, x, y):
        for key, val in self.state.items():
            if "_" in key:
                continue
            if (val == (x, y)).all():
                return key
        return None

    def step(self, action):
        assert action < self.action_dim

        prev_pos = self.pos
        prev_dir = self.dir
        prev_state = copy.deepcopy(self.state)
        state = self.state
        inventory = self.inventory
        cookbook = self.cookbook

        remove_thing_from_grid = crafted_in_workshop = start_craft_in_furnace = False
        thing = thing_name = facing_change_name = None

        # move actions
        if action in self.move_target_names:
            target_name = self.move_target_names[action]
            if target_name in ["workshop", "furnace"] or not state[target_name + "_picked"]:
                neighbors = self.neighbors(state[target_name])
                if target_name in ["workshop", "furnace"]:
                    random.shuffle(neighbors)
                for x, y, dir in neighbors:
                    if not self.grid[x, y].any():
                        self.pos = (x, y)
                        self.dir = opposite_dir[dir]
                        cur_facing = [k for k, v in state.items() if k.endswith("_faced") and v]
                        if cur_facing:
                            assert len(cur_facing) == 1
                            state[cur_facing[0]] = False
                        state[target_name + "_faced"] = True
                        break
        else:
            for nx, ny, _ in self.neighbors(self.pos, self.dir):
                here = self.grid[nx, ny, :]
                if not self.grid[nx, ny, :].any():
                    continue

                assert here.sum() == 1, "impossible world configuration"
                thing = here.argmax()
                if thing == self.boundary_idx:
                    continue

                thing_name = self.find_key(nx, ny)
                assert state[thing_name + "_faced"]

                if action == USE:
                    if thing in self.primitive_idxes:
                        primitive_info = self.primitives[thing]
                        required_tool_idx = primitive_info.get("_require", None)
                        if required_tool_idx is None or inventory[required_tool_idx] > 0:
                            remove_thing_from_grid = True
                            inventory[thing] += 1
                    elif thing == self.workshop_idx:
                        continue
                    elif thing == self.furnace_idx:
                        furnace_slot, furnace_stage = state["furnace_slot"], state["furnace_stage"]
                        if not state["furnace_ready"]:
                            if inventory[self.furnace_idx]:
                                assert furnace_slot == furnace_stage == 0
                                state["furnace_ready"] = True
                                inventory[self.furnace_idx] -= 1
                        else:
                            if furnace_slot != 0:
                                craft_idx = cookbook.furnace_slot2idx[furnace_slot]
                                craft_recipe = self.recipes[craft_idx]
                                if furnace_stage == craft_recipe["_step"]:
                                    inventory[craft_idx] += craft_recipe.get("_yield", 1)
                                    state["furnace_slot"], state["furnace_stage"] = 0, 0
                    else:
                        env_obj_info = self.environments[thing]
                        required_tool_idx = env_obj_info.get("_require", None)
                        if required_tool_idx is None or inventory[required_tool_idx] > 0:
                            remove_thing_from_grid = True
                            if env_obj_info["_consume"]:
                                inventory[required_tool_idx] -= 1

                    if remove_thing_from_grid:
                        self.grid[nx, ny, thing] = 0
                        state[thing_name] = np.array([0, 0])
                        state[thing_name + "_picked"] = True
                        state[thing_name + "_faced"] = False

                    break

                else:
                    output = self.craft_idxes[action - self.craft_action_starts]
                    recipe = self.recipes[output]
                    if thing != recipe["_at"]:
                        continue

                    yld = recipe.get("_yield", 1)
                    ing = [i for i in recipe if isinstance(i, int)]
                    if any(inventory[i] < recipe[i] for i in ing):
                        continue

                    if thing == self.workshop_idx:
                        for i in ing:
                            inventory[i] -= recipe[i]
                        inventory[output] += yld
                        crafted_in_workshop = True
                    elif thing == self.furnace_idx:
                        if not state["furnace_ready"] or state["furnace_slot"] != 0:
                            continue
                        for i in ing:
                            inventory[i] -= recipe[i]
                        state["furnace_slot"] = cookbook.idx2furnace_slot[output]
                        start_craft_in_furnace = True
                    else:
                        raise NotImplementedError

        if self.has_furnace:
            furnace_slot, furnace_stage = state["furnace_slot"], state["furnace_stage"]
            if furnace_slot != 0 and not start_craft_in_furnace:
                assert state["furnace_ready"]
                craft_idx = cookbook.furnace_slot2idx[furnace_slot]
                if furnace_stage < self.recipes[craft_idx]["_step"]:
                    state["furnace_stage"] += 1

        evaluate_mask = True
        if evaluate_mask:
            if self.slice_dict is None:
                slice_dict, cum = {}, 0
                for k in self.dynamics_keys:
                    k_dim = len(self.obs_dims[k])
                    slice_dict[k] = slice(cum, cum + k_dim)
                    cum += k_dim
                self.slice_dict, self.feature_dim = slice_dict, cum
            else:
                slice_dict = self.slice_dict
            action_idx = self.feature_dim

            mask = np.eye(self.feature_dim, self.feature_dim + 1, dtype=bool)
            inventory_offset = slice_dict["inventory"].start

            if self.has_furnace:
                furnace_ready_slice = slice_dict["furnace_ready"]
                furnace_slot_slice = slice_dict["furnace_slot"]
                furnace_stage_slice = slice_dict["furnace_stage"]

            if self.use_pose:
                agent_pos_slice, agent_dir_slice = slice_dict["agent_pos"], slice_dict["agent_dir"]

                if action in self.move_target_names:
                    target_pos_slice = slice_dict[target_name]
                    if self.pos != prev_pos or self.dir != prev_dir:
                        for slice_ in [agent_pos_slice, agent_dir_slice]:
                            mask[slice_, slice_] = False
                            mask[slice_, target_pos_slice] = True
                            mask[slice_, action_idx] = True
                else:
                    if thing_name is not None:
                        thing_pos_slice = slice_dict[thing_name]

                    if action == USE:
                        if remove_thing_from_grid:
                            mask[thing_pos_slice, agent_pos_slice] = True
                            mask[thing_pos_slice, agent_dir_slice] = True
                            mask[thing_pos_slice, thing_pos_slice] = True
                            mask[thing_pos_slice, action_idx] = True

                            if thing in self.primitive_idxes:
                                thing_invent_idx = inventory_offset + thing
                                mask[thing_invent_idx, agent_pos_slice] = True
                                mask[thing_invent_idx, agent_dir_slice] = True
                                mask[thing_invent_idx, thing_pos_slice] = True
                                mask[thing_invent_idx, action_idx] = True
                                if required_tool_idx:
                                    tool_invent_idx = inventory_offset + required_tool_idx
                                    mask[thing_pos_slice, tool_invent_idx] = True
                                    mask[thing_invent_idx, tool_invent_idx] = True
                            else:
                                if required_tool_idx:
                                    tool_invent_idx = inventory_offset + required_tool_idx
                                    mask[thing_pos_slice, tool_invent_idx] = True
                                    if env_obj_info["_consume"]:
                                        mask[tool_invent_idx, agent_pos_slice] = True
                                        mask[tool_invent_idx, agent_dir_slice] = True
                                        mask[tool_invent_idx, thing_pos_slice] = True
                                        mask[tool_invent_idx, tool_invent_idx] = True
                                        mask[tool_invent_idx, action_idx] = True

                        if self.has_furnace:
                            if prev_state["furnace_ready"] != state["furnace_ready"]:
                                furnace_invent_idx = inventory_offset + self.furnace_idx
                                mask[furnace_ready_slice, agent_pos_slice] = True
                                mask[furnace_ready_slice, agent_dir_slice] = True
                                mask[furnace_ready_slice, thing_pos_slice] = True
                                mask[furnace_ready_slice, furnace_invent_idx] = True
                                mask[furnace_ready_slice, action_idx] = True

                            if prev_state["furnace_slot"] != state["furnace_slot"]:
                                craft_invent_idx = inventory_offset + craft_idx
                                for slice_ in [furnace_slot_slice, furnace_stage_slice, craft_invent_idx]:
                                    mask[slice_, agent_pos_slice] = True
                                    mask[slice_, agent_dir_slice] = True
                                    mask[slice_, thing_pos_slice] = True
                                    mask[slice_, furnace_slot_slice] = True
                                    mask[slice_, furnace_stage_slice] = True
                                    mask[slice_, action_idx] = True

                    else:
                        if crafted_in_workshop or start_craft_in_furnace:
                            ing = [inventory_offset + i for i in ing]
                            if thing == self.furnace_idx:
                                things = ing + [slice_dict["furnace_slot"]]
                                ing = ing + [slice_dict["furnace_ready"], slice_dict["furnace_slot"]]
                            else:
                                things = ing + [inventory_offset + output]
                            for thing in things:
                                mask[thing, agent_pos_slice] = True
                                mask[thing, agent_dir_slice] = True
                                mask[thing, thing_pos_slice] = True
                                mask[thing, action_idx] = True
                                for i in ing:
                                    mask[thing, i] = True

                if self.has_furnace and state["furnace_stage"] != prev_state["furnace_stage"]:
                    mask[furnace_stage_slice, furnace_slot_slice] = True
                    mask[furnace_stage_slice, furnace_stage_slice] = True

            else:
                raise NotImplementedError
                if action == USE:
                    if remove_thing_from_grid:
                        picked_slice = slice_dict[thing_name + "_picked"]
                        faced_slice = slice_dict[thing_name + "_faced"]
                        mask[picked_slice, faced_slice] = True
                        mask[picked_slice, action_idx] = True
                        mask[faced_slice, action_idx] = True
                        if thing in self.primitive_idxes:
                            thing_invent_idx = inventory_offset + thing
                            mask[thing_invent_idx, faced_slice] = True
                            mask[thing_invent_idx, action_idx] = True

                        elif thing == self.water_idx:
                            bridge_invent_idx = inventory_offset + cookbook.index["bridge"]
                            mask[picked_slice, bridge_invent_idx] = True
                            mask[bridge_invent_idx, faced_slice] = True
                            mask[bridge_invent_idx, action_idx] = True

                        elif thing == self.stone_index:
                            axe_invent_idx = inventory_offset + cookbook.index["axe"]
                    elif thing == self.furnace_idx:
                        mask[picked_slice, axe_invent_idx] = True
                elif action in self.move_target_names:
                    faced_slice = slice_dict[target_name + "_faced"]
                    if state[target_name + "_faced"]:
                        mask[faced_slice, action_idx] = True
                        if facing_change_name is not None:
                            mask[slice_dict[facing_change_name], action_idx] = True
                    else:
                        picked_slice = slice_dict[target_name + "_picked"]
                        mask[faced_slice, picked_slice] = True
                else:
                    if craft_success:
                        for thing in [output] + ing:
                            thing_invent_idx = inventory_offset + thing
                            mask[thing_invent_idx, slice_dict["workshop_faced"]] = True
                            mask[thing_invent_idx, action_idx] = True
                            for i in ing:
                                mask[thing_invent_idx, inventory_offset + i] = True

        self.cur_step += 1
        done = self.cur_step >= self.horizon

        reward = self.reward()
        info = {"success": self.check_success(),
                "stage_completion": self.stage_completion_tracker}
        if evaluate_mask:
            info["local_causality"] = mask

        return self.get_state(), reward, done, info

    def next_to(self, i_kind):
        x, y = self.pos
        return self.grid[x - 1:x + 2, y - 1:y + 2, i_kind].any()

    def render(self):
        h, w = self.height, self.width
        cell_w = 3

        # First row
        print(" " * (cell_w + 1), end='')
        for i in range(w):
            print("| {:^{}d} ".format(i, cell_w), end='')
        print("| ")
        print((w * (cell_w + 3) + cell_w + 2) * "-")

        # Other rows
        for j in reversed(range(h)):
            print("{:{}d} ".format(j, cell_w), end='')

            for i in range(w):
                symbol = ""
                if (i, j) == self.pos:
                    if self.dir == LEFT:
                        symbol = u"\u2190"
                    elif self.dir == RIGHT:
                        symbol = u"\u2192"
                    elif self.dir == UP:
                        symbol = u"\u2191"
                    elif self.dir == DOWN:
                        symbol = u"\u2193"
                elif self.grid[i, j].any():
                    thing = self.grid[i, j].argmax()
                    name = self.cookbook.index[thing]
                    state_key = self.find_key(i, j)
                    if thing in self.primitive_idxes and state_key and state_key[-1].isdigit():
                        symbol = name[:cell_w - 1] + state_key[-1]
                    else:
                        symbol = name[:cell_w]

                assert len(symbol) <= cell_w

                print("| {:^{}} ".format(symbol, cell_w), end='')
            print("| ")
            print((w * (cell_w + 3) + cell_w + 2) * "-")

        print("inventory")
        for i in range(len(self.inventory)):
            if i == self.furnace_idx or i in self.primitive_idxes + self.craft_idxes:
                print("| {} ".format(self.cookbook.index[i]), end='')
        print("| ")
        for i, num in enumerate(self.inventory):
            if i == self.furnace_idx or i in self.primitive_idxes + self.craft_idxes:
                print("| {:^{}} ".format(num, len(self.cookbook.index[i])), end='')
        print("| ")

        if self.has_furnace:
            print("furnace")
            slot_msg = step_msg = ""
            slot = self.state["furnace_slot"]
            if slot:
                craft_idx = self.cookbook.furnace_slot2idx[slot]
                slot_msg = self.cookbook.index[craft_idx]
                total_stage = self.recipes[craft_idx]["_step"]
                step_msg = "{}/{}".format(self.state["furnace_stage"], total_stage)

            msgs = [["ready", "Y" if self.state["furnace_ready"] else "N"],
                    ["slot", slot_msg],
                    ["step", step_msg]]
            for i in range(len(msgs[0])):
                for ele in msgs:
                    ele_w = max([len(e) for e in ele])
                    print("| {:^{}} ".format(ele[i], ele_w), end='')
                print("| ")
            print()



