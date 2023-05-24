from env.cookbook import Cookbook

import os
import copy
import numpy as np


DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3
USE = 4


def random_free(grid):
    pos = None
    while pos is None:
        x, y = np.random.randint(grid.shape[0]), np.random.randint(grid.shape[1])
        if grid[x, y, :].any():
            continue
        pos = (x, y)
    return pos


class CraftWorld(object):
    def __init__(self, params):
        self.params = params
        self.env_params = env_params = params.env_params
        self.use_stage_reward = env_params.use_stage_reward
        self.craft_env_params = craft_env_params = env_params.craft_env_params

        self.goal = craft_env_params.goal
        self.width = craft_env_params.width
        self.height = craft_env_params.height
        self.num_ingredients = craft_env_params.num_ingredients
        self.horizon = craft_env_params.horizon

        recipe_path = os.path.join((os.path.dirname(os.path.realpath(__file__))), "craft_recipes", craft_env_params.recipe_fname)
        self.cookbook = Cookbook(recipe_path)

        self.goal_idx = self.cookbook.index[self.goal]
        self.non_grabbable_indices = self.cookbook.environment
        self.primitive_indices = self.cookbook.primitives
        self.creatable_indices = [i for i in range(self.cookbook.n_kinds)
                if i not in self.non_grabbable_indices + self.primitive_indices]
        self.workshop_index = self.cookbook.index["workshop"]
        self.water_index = self.cookbook.index["water"]
        self.stone_index = self.cookbook.index["stone"]

        self.action_dim = 5 + len(self.creatable_indices)

        # for computing staged reward
        self.treasure_name = self.goal
        if self.treasure_name == "gold":
            self.craft_tool = "bridge"
        elif self.treasure_name == "gem":
            self.craft_tool = "axe"
        else:
            raise NotImplementedError
        self.craft_tool_index = self.cookbook.index[self.craft_tool]
        self.craft_tool_recipe, self.num_craft_tool_stages = self.add_recipe(self.craft_tool_index)

        self.reset()

    def add_recipe(self, index):
        recipe_new, num_stages = {}, 0
        recipe = self.cookbook.recipes[index]
        for k, v in recipe.items():
            if k in ["_at", "_yield"]:
                continue
            if k in self.primitive_indices:
                recipe_new[k] = v
            else:
                assert k in self.cookbook.recipes
                recipe, num_ingredient_stages = self.add_recipe(k)
                recipe_new[k] = [v, recipe, num_ingredient_stages]
                num_stages += v * num_ingredient_stages
            num_stages += v
        return recipe_new, num_stages

    def reset(self):
        self.cur_step = 0
        goal = self.goal_idx

        assert goal not in self.cookbook.environment
        if goal in self.primitive_indices:
            make_island = goal == self.cookbook.index["gold"]
            make_cave = goal == self.cookbook.index["gem"]
            self.sample_scenario({goal: 1}, make_island=make_island, make_cave=make_cave)
        elif goal in self.cookbook.recipes:
            ingredients = self.cookbook.primitives_for(goal)
            self.sample_scenario(ingredients)
        else:
            assert False, "don't know how to build a scenario for %s" % goal

        self.stage_completion_tracker = 0

        return self.get_state()

    def observation_spec(self):
        return self.get_state()

    def observation_dims(self):
        state = {}
        for key in self.state:
            state[key] = np.array([self.width + 1, self.height + 1])

        state["agent_pos"] = np.array([self.width + 1, self.height + 1])
        state["agent_dir"] = np.array([4])
        inventory = self.inventory[len(self.non_grabbable_indices):]
        state["inventory"] = np.ones_like(inventory, dtype=int) * (self.num_ingredients + 1)
        state["step_count"] = np.array([1])

        return state

    def sample_scenario(self, ingredients, make_island=False, make_cave=False):
        # generate grid
        grid = np.zeros((self.width, self.height, self.cookbook.n_kinds))
        # i_bd = self.cookbook.index["boundary"]
        # grid[0, :, i_bd] = 1
        # grid[self.width - 1:, :, i_bd] = 1
        # grid[:, 0, i_bd] = 1
        # grid[:, self.height - 1:, i_bd] = 1

        self.state = {}

        # treasure
        if make_island or make_cave:
            gx, gy = np.random.randint(1, self.width - 1), 0
            self.treasure_index = self.cookbook.index[self.treasure_name]
            wall_index = self.water_index if make_island else self.stone_index
            grid[gx, gy, self.treasure_index] = 1
            self.state[self.treasure_name] = np.array([gx, gy])

            num_walls = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= gx + i < self.width and 0 <= gy + j < self.height and not grid[gx + i, gy + j, :].any():
                        grid[gx + i, gy + j, wall_index] = 1
                        self.state["wall{}".format(num_walls)] = np.array([gx + i, gy + j])
                        num_walls += 1

        # ingredients
        for primitive in self.primitive_indices:
            if primitive == self.cookbook.index["gold"] or primitive == self.cookbook.index["gem"]:
                continue
            for i in range(self.num_ingredients):
                x, y = random_free(grid)
                grid[x, y, primitive] = 1
                item = self.cookbook.index.reverse_contents[primitive]
                self.state["{}{}".format(item, i)] = np.array([x, y])

        # generate crafting stations
        ws_x, ws_y = random_free(grid)
        grid[ws_x, ws_y, self.workshop_index] = 1
        self.state["workshop"] = np.array([ws_x, ws_y])

        self.grid = grid

        # generate init pos
        self.pos = random_free(grid)
        self.dir = 0
        self.inventory = np.zeros(self.cookbook.n_kinds, dtype=int)

    def get_state(self):
        state = copy.deepcopy(self.state)
        state["agent_pos"] = np.array(self.pos)
        state["agent_dir"] = np.array([self.dir])
        state["inventory"] = self.inventory[len(self.non_grabbable_indices):].copy()
        state["step_count"] = np.array([float(self.cur_step) / self.horizon])

        return state

    def can_collect_treasure(self):
        has_craft_tool = self.inventory[self.craft_tool_index] > 0

        if self.treasure_name == "gem":
            return has_craft_tool
        elif self.treasure_name == "gold":
            if has_craft_tool:
                return True
            grid = self.grid
            treasure_x, treasure_y = self.state[self.treasure_name]
            has_path = False
            for dx, dy in [[1, 0], [-1, 0], [0, 1]]:
                if not grid[treasure_x + dx, treasure_y + dy].any():
                    has_path = True
            return has_path
        else:
            raise NotImplementedError

    def check_craft_tool_completion(self, recipe, inventory=None, reward_scale=1.0):
        if inventory is None:
            inventory = self.inventory.copy()

        completion_reward = 0
        item_reward = reward_scale / len(recipe)
        for k, num in recipe.items():
            if k in self.primitive_indices:
                if inventory[k] >= num:
                    inventory[k] -= num
                    completion_reward += item_reward
            elif k in self.cookbook.recipes:
                num, k_recipe, num_k_stages = num
                self.stage_completion_tracker += min(num, inventory[k]) * num_k_stages
                if inventory[k] >= num:
                    inventory[k] -= num
                    completion_reward += item_reward
                else:
                    completion_reward += self.check_craft_tool_completion(k_recipe, inventory, item_reward / 2)
            else:
                raise NotImplementedError
            self.stage_completion_tracker += min(num, inventory[k])
        return completion_reward

    def reward(self):
        if self.check_success():
            reward = 1
            self.stage_completion_tracker = self.num_craft_tool_stages + 2
        elif self.can_collect_treasure():
            reward = 0.75
            self.stage_completion_tracker = self.num_craft_tool_stages + 1
        else:
            self.stage_completion_tracker = 0
            reward = self.check_craft_tool_completion(self.craft_tool_recipe, reward_scale=0.5)

        if self.use_stage_reward:
            return reward
        else:
            return float(reward == 1)

    def check_success(self,):
        return self.inventory[self.goal_idx] > 0

    def neighbors(self, pos, dir=None):
        x, y = pos
        neighbors = []
        if x > 0 and (dir is None or dir == LEFT):
            neighbors.append((x-1, y))
        if y > 0 and (dir is None or dir == DOWN):
            neighbors.append((x, y-1))
        if x < self.width - 1 and (dir is None or dir == RIGHT):
            neighbors.append((x+1, y))
        if y < self.height - 1 and (dir is None or dir == UP):
            neighbors.append((x, y+1))
        return neighbors

    def find_key(self, x, y):
        assert self.grid[x, y, :].any()
        item = self.cookbook.index.reverse_contents[self.grid[x, y, :].argmax()]
        for key, val in self.state.items():
            if (val == (x, y)).all():
                return key
        raise NotImplementedError("no key found at ({}, {})".format(x, y))

    def step(self, action):
        assert action < self.action_dim

        x, y = self.pos
        n_dir = self.dir
        n_inventory = self.inventory
        n_grid = self.grid

        remove_thing_from_grid = has_obstacle = craft_success = False

        # move actions
        if action == DOWN:
            dx, dy = (0, -1)
            n_dir = DOWN
        elif action == UP:
            dx, dy = (0, 1)
            n_dir = UP
        elif action == LEFT:
            dx, dy = (-1, 0)
            n_dir = LEFT
        elif action == RIGHT:
            dx, dy = (1, 0)
            n_dir = RIGHT
        # use or craft actions
        else:
            cookbook = self.cookbook
            dx, dy = 0, 0

            for nx, ny in self.neighbors(self.pos, self.dir):
                here = self.grid[nx, ny, :]
                if not self.grid[nx, ny, :].any():
                    continue

                assert here.sum() == 1, "impossible world configuration"
                thing = here.argmax()

                if not (thing in self.primitive_indices or \
                        thing == self.workshop_index or \
                        thing == self.water_index or \
                        thing == self.stone_index):
                    continue

                if action == USE:

                    if thing == self.workshop_index:
                        continue

                    if thing in self.primitive_indices:
                        n_inventory[thing] += 1
                        remove_thing_from_grid = True

                    elif thing == self.water_index:
                        if n_inventory[cookbook.index["bridge"]] > 0:
                            n_inventory[cookbook.index["bridge"]] -= 1
                            remove_thing_from_grid = True

                    elif thing == self.stone_index:
                        if n_inventory[cookbook.index["axe"]] > 0:
                            remove_thing_from_grid = True

                    else:
                        raise NotImplementedError

                    if remove_thing_from_grid:
                        key = self.find_key(nx, ny)
                        n_grid[nx, ny, thing] = 0
                        self.state[key] = np.array([self.width, self.height])

                    break
                else:
                    if thing != self.workshop_index:
                        continue

                    output = self.creatable_indices[action - USE - 1]
                    workshop = cookbook.index.reverse_contents[thing]

                    recipe = cookbook.recipes[output]
                    if recipe["_at"] != workshop:
                        continue
                    yld = recipe["_yield"] if "_yield" in recipe else 1
                    ing = [i for i in recipe if isinstance(i, int)]
                    if any(n_inventory[i] < recipe[i] for i in ing):
                        continue
                    n_inventory[output] += yld
                    for i in ing:
                        n_inventory[i] -= recipe[i]
                    craft_success = True
                    break

        n_x = x + dx
        n_y = y + dy
        if not 0 <= n_x < self.width:
            n_x = x
        if not 0 <= n_y < self.height:
            n_y = y
        if self.grid[n_x, n_y, :].any():
            has_obstacle = True
            n_x, n_y = x, y
        self.pos = (n_x, n_y)
        self.dir = n_dir

        evaluate_mask = True
        if evaluate_mask:
            slice_dict, cum = {}, 0
            for k in self.params.dynamics_keys:
                k_dim = len(self.params.obs_dims[k])
                slice_dict[k] = slice(cum, cum + k_dim)
                cum += k_dim
            action_idx = cum
            mask = np.eye(cum, cum + 1, dtype=bool)
            inventory_offset = slice_dict["inventory"].start - len(self.non_grabbable_indices)

            if action in [DOWN, UP, LEFT, RIGHT]:
                pos_idx = int(action < 2)
                mask[slice_dict["agent_dir"], action_idx] = True
                mask[pos_idx, action_idx] = True
                if (x, y) == self.pos:
                    mask[pos_idx, slice_dict["agent_pos"]] = True
                    if has_obstacle:
                        n_x, n_y = x + dx, y + dy
                        obstacle_key = self.find_key(n_x, n_y)
                        mask[pos_idx, slice_dict[obstacle_key]] = True
            elif action == USE:
                if remove_thing_from_grid:
                    thing_slice = slice_dict[key]
                    mask[thing_slice, slice_dict["agent_dir"]] = True
                    mask[thing_slice, slice_dict["agent_pos"]] = True
                    mask[thing_slice, thing_slice] = True
                    mask[thing_slice, action_idx] = True
                    if thing in self.primitive_indices:
                        thing_invent_idx = inventory_offset + thing
                        mask[thing_invent_idx, slice_dict["agent_dir"]] = True
                        mask[thing_invent_idx, slice_dict["agent_pos"]] = True
                        mask[thing_invent_idx, thing_slice] = True
                        mask[thing_invent_idx, action_idx] = True

                    elif thing == self.water_index:
                        bridge_invent_idx = inventory_offset + cookbook.index["bridge"]
                        mask[thing_slice, bridge_invent_idx] = True
                        mask[bridge_invent_idx, slice_dict["agent_dir"]] = True
                        mask[bridge_invent_idx, slice_dict["agent_pos"]] = True
                        mask[bridge_invent_idx, thing_slice] = True
                        mask[bridge_invent_idx, action_idx] = True

                    elif thing == self.stone_index:
                        axe_invent_idx = inventory_offset + cookbook.index["axe"]
                        mask[thing_slice, axe_invent_idx] = True
            else:
                if craft_success:
                    for thing in [output] + ing:
                        thing_invent_idx = inventory_offset + thing
                        mask[thing_invent_idx, slice_dict["agent_dir"]] = True
                        mask[thing_invent_idx, slice_dict["agent_pos"]] = True
                        mask[thing_invent_idx, slice_dict["workshop"]] = True
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
