import random
import numpy as np

TREASURE_NAMES = ["gold", "gem"]

DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3
USE = 4

class ScriptedCraft:
    def __init__(self, env, params):
        self.env = env

        scripted_policy_params = params.scripted_policy_params
        craft_params = scripted_policy_params.craft_params
        self.random_action_prob = scripted_policy_params.random_action_prob
        self.random_primitive_prob = craft_params.random_primitive_prob
        self.random_craft_prob = craft_params.random_craft_prob

        self.treasure_name = params.env_params.craft_env_params.goal

        if self.treasure_name == "gold":
            self.craft_tool = "bridge"
        elif self.treasure_name == "gem":
            self.craft_tool = "axe"
        else:
            raise NotImplementedError

        self.cookbook = cookbook = env.cookbook
        self.primitives = cookbook.primitives
        craft_tool_index = cookbook.index[self.craft_tool]
        self.craft_tool_recipe = self.add_recipe(craft_tool_index)

        self.STATES = ["COLLECT_PRIMITIVE", "CRAFT", "COLLECT_TREASURE"]
        self.reset()

    def reset(self, *args):
        self.reset_state_machine()

    def reset_state_machine(self):
        self.state = "COLLECT_PRIMITIVE"
        self.mov_goal = None
        self.craft_goal = None
        self.cur_state_step = 0

    def add_recipe(self, index):
        recipe_new = {}
        recipe = self.cookbook.recipes[index]
        for k, v in recipe.items():
            if k in ["_at", "_yield"]:
                continue
            if k in self.primitives:
                recipe_new[k] = v
            else:
                assert k in self.cookbook.recipes
                recipe_new[k] = [v, self.add_recipe(k)]
        return recipe_new

    def find_missing_primitives(self, recipe, inventory=None, return_set=None):
        if return_set is None:
            inventory = self.env.inventory.copy()
            return_set = set()
        for k, v in recipe.items():
            if k in self.primitives:
                if inventory[k] >= v:
                    inventory[k] -= v
                else:
                    return_set.add(k)
            elif k in self.cookbook.recipes:
                num, k_recipe = v
                return_set = self.find_missing_primitives(k_recipe, inventory, return_set)
            else:
                raise NotImplementedError
        return return_set

    def can_craft(self):
        return not self.find_missing_primitives(self.craft_tool_recipe)

    def has_treasure(self):
        treasure_index = self.cookbook.index[self.treasure_name]
        has_treasure = self.env.inventory[treasure_index] > 0
        return has_treasure

    def can_collect_treasure(self):
        env = self.env

        craft_tool_index = self.cookbook.index[self.craft_tool]
        has_craft_tool = env.inventory[craft_tool_index] > 0

        if self.treasure_name == "gem":
            return has_craft_tool
        elif self.treasure_name == "gold":
            if has_craft_tool:
                return True
            grid = env.grid
            treasure_x, treasure_y = env.state[self.treasure_name]
            has_path = False
            for dx, dy in [[1, 0], [-1, 0], [0, 1]]:
                if not grid[treasure_x + dx, treasure_y + dy].any():
                    has_path = True
            return has_path
        else:
            raise NotImplementedError

    def update_state_machine(self):
        state_changed = False
        prev_state = self.state
        if self.state == "COLLECT_PRIMITIVE":
            if self.mov_goal is None or (self.env.pos == self.mov_goal).all():
                if self.can_craft():
                    self.state = "CRAFT"
                else:
                    prev_state = None
        elif self.state == "CRAFT":
            if not self.has_treasure() and self.can_collect_treasure():
                self.state = "COLLECT_TREASURE"
            if not self.can_craft():
                self.state = "COLLECT_PRIMITIVE"
        elif self.state == "COLLECT_TREASURE":
            if self.has_treasure():
                self.state = "COLLECT_PRIMITIVE"
            elif not self.can_collect_treasure():
                if self.can_craft():
                    self.state = "CRAFT"
                else:
                    self.state = "COLLECT_PRIMITIVE"
        else:
            raise NotImplementedError

        if prev_state != self.state or self.cur_state_step >= 30:
            self.cur_state_step = 0
            self.update_goal()

    def pick_primitive(self):
        env = self.env
        state = env.state
        inventory = env.inventory
        num_ingredients = env.num_ingredients
        width, height = env.width, env.height

        cookbook = self.cookbook
        primitive_names = cookbook.primitive_names

        all_available_primitives = []
        needed_primitives = []
        missing_primitive_indices = self.find_missing_primitives(self.craft_tool_recipe)
        for primitive_name in primitive_names:
            primitive_index = cookbook.index[primitive_name]
            for i in range(num_ingredients):
                if primitive_name in ["goal", "gem"]:
                    key = primitive_name
                else:
                    key = "{}{}".format(primitive_name, i)
                if key not in state or (state[key] == (width, height)).all():
                    continue
                all_available_primitives.append(key)
                if primitive_index in missing_primitive_indices:
                    needed_primitives.append(key)

        if np.random.rand() < self.random_primitive_prob or self.has_treasure():
            primitives_candidates = all_available_primitives
        else:
            primitives_candidates = needed_primitives

        if not primitives_candidates:
            return np.random.randint((width, height))
        else:
            primitive = random.choice(primitives_candidates)
            return state[primitive]

    def select_craft(self, output_index, recipe):
        for k, v in recipe.items():
            if k in self.primitives:
                assert self.env.inventory[k] >= v
            elif k in self.cookbook.recipes:
                num, k_recipe = v
                if self.env.inventory[k] < num:
                    return self.select_craft(k, k_recipe)
            else:
                raise NotImplementedError
        return output_index

    def pick_craft(self):
        self.mov_goal = self.env.state["workshop"]
        cookbook = self.cookbook
        if np.random.rand() < self.random_craft_prob:
            return random.choice(list(cookbook.recipes.keys()))
        else:
            craft_tool_index = cookbook.index[self.craft_tool]
            return self.select_craft(craft_tool_index, self.craft_tool_recipe)

    def update_goal(self):
        env = self.env
        if self.state == "COLLECT_PRIMITIVE":
            self.mov_goal = self.pick_primitive()
        elif self.state == "CRAFT":
            self.mov_goal = env.state["workshop"]
            self.craft_goal = self.pick_craft()
        elif self.state == "COLLECT_TREASURE":
            self.mov_goal = env.state[env.treasure_name]
        else:
            raise NotImplementedError

    def act(self, obs):
        env = self.env

        if np.random.rand() < self.random_action_prob:
            if self.state == "CRAFT":
                return np.random.randint(env.action_dim)
            else:
                return np.random.randint(USE + 1)

        self.update_state_machine()

        grid = env.grid
        x, y = env.pos
        dir = env.dir
        mov_x, mov_y = self.mov_goal

        dirs = []
        if x < mov_x:
            dirs.append(RIGHT)
        if x > mov_x:
            dirs.append(LEFT)
        if y < mov_y:
            dirs.append(UP)
        if y > mov_y:
            dirs.append(DOWN)

        action = None
        if dir in dirs:
            neighbors = env.neighbors((x, y), dir)
            fwd_x, fwd_y = neighbors[0]
            fwd_cell = grid[fwd_x, fwd_y, :]
            if fwd_cell.any():
                thing = fwd_cell.argmax()

                if self.state == "CRAFT" and thing == env.workshop_index:
                    action = env.creatable_indices.index(self.craft_goal) + USE + 1
                if self.state != "CRAFT" and thing != env.workshop_index:
                    action = USE

        if action is None:
            if len(dirs):
                action = random.choice(dirs)
            else:
                action = np.random.randint(env.action_dim)

        self.cur_state_step += 1

        return action
