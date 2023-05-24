import random
import numpy as np

USE = 0


class ScriptedCraft:
    def __init__(self, env, params):
        self.env = env

        scripted_policy_params = params.scripted_policy_params
        craft_params = scripted_policy_params.craft_params
        self.random_action_prob = scripted_policy_params.random_action_prob
        self.random_primitive_prob = craft_params.random_primitive_prob
        self.random_craft_prob = craft_params.random_craft_prob

        self.goal = env.goal
        self.goal_idx = env.goal_idx

        self.cookbook = cookbook = env.cookbook
        self.environment_idxes = cookbook.environment_idxes
        self.primitive_idxes = cookbook.primitive_idxes
        self.craft_idxes = cookbook.craft_idxes

        self.environments = cookbook.environments
        self.primitives = cookbook.primitives
        self.recipes = cookbook.recipes

        self.workshop_idx = env.workshop_idx
        self.furnace_idx = env.furnace_idx

        self.craft_tools = env.craft_tools
        self.inter_tools = env.inter_tools

        self.STATES = ["COLLECT", "CRAFT", "SETUP_FURNACE", "COLLECT_TREASURE"]
        self.reset()

    def reset(self, *args):
        self.reset_state_machine()

    def reset_state_machine(self):
        self.state = "COLLECT"
        self.collect_goal = None
        self.craft_goal = None
        self.num_collect_steps = 0

    def update_state_machine(self):
        prev_state = self.state
        if self.state == "COLLECT":
            if self.craft_goal is None:
                prev_state = None
            elif self.can_craft(self.craft_goal):
                self.state = "CRAFT"
            elif self.collect_goal is None:
                prev_state = None
            elif self.env.state[self.collect_goal + "_picked"] or self.num_collect_steps >= 5:
                self.num_collect_steps = 0
                if self.can_craft(self.craft_goal):
                    self.state = "CRAFT"
                else:
                    self.collect_goal = self.pick_primitive(self.craft_goal)
        elif self.state == "CRAFT":
            station_idx = self.recipes[self.craft_goal]["_at"]
            if self.env.inventory[self.craft_goal]:
                if self.craft_goal == self.furnace_idx:
                    self.state = "SETUP_FURNACE"
                elif not self.has_treasure() and self.can_collect_treasure():
                    self.state = "COLLECT_TREASURE"
                else:
                    self.state = "COLLECT"
            elif not self.can_craft(self.craft_goal) and not (station_idx == self.furnace_idx and self.in_furnace(self.craft_goal)):
                self.state = "COLLECT"
        elif self.state == "SETUP_FURNACE":
            if self.env.state["furnace_ready"]:
                self.state = "COLLECT"
        elif self.state == "COLLECT_TREASURE":
            if self.has_treasure():
                self.state = "COLLECT"
            elif not self.can_collect_treasure():
                self.state = "COLLECT"
        else:
            raise NotImplementedError

        assert self.state in self.STATES

        if prev_state != self.state and self.state == "COLLECT":
            self.num_collect_steps = 0
            self.craft_goal = self.pick_craft()
            self.collect_goal = self.pick_primitive(self.craft_goal)

    def pick_craft(self):
        craft_idx = None

        # pick a tool necessary for collecting treasure
        if np.random.rand() >= self.random_craft_prob:
            # this "|" merges two dicts but need python >= 3.9
            for k, (recipe, _) in (self.inter_tools | self.craft_tools).items():
                if not self.has_craft(k) and self.possible_to_craft(k):
                    craft_idx = self.select_craft(k)
                    break

        while craft_idx is None:
            idx = random.choice(self.craft_idxes)
            if self.possible_to_craft(idx):
                craft_idx = idx

        return craft_idx

    def pick_primitive(self, craft_idx):
        env = self.env
        state = env.state
        grid = env.grid
        inventory = env.inventory
        cookbook = env.cookbook

        all_available_primitives = []
        needed_primitives = []
        missing_primitive_idxes = self.find_missing_primitives(craft_idx)
        for thing_name, thing_pos in state.items():
            if "_" in thing_name:
                continue

            x, y = thing_pos
            thing_idx = grid[x, y].argmax()
            if thing_name in ["workshop", "furnace"] or state[thing_name + "_picked"]:
                continue

            all_available_primitives.append(thing_name)
            if thing_idx in missing_primitive_idxes:
                needed_primitives.append(thing_name)

        if np.random.rand() < self.random_primitive_prob or self.has_treasure():
            primitives_candidates = all_available_primitives
        else:
            primitives_candidates = needed_primitives

        if not primitives_candidates:
            return None
        else:
            return random.choice(primitives_candidates)

    def find_missing_primitives(self, craft_idx, inventory=None, return_set=None):
        if return_set is None:
            inventory = self.env.inventory.copy()
            return_set = set()
        for k, v in self.recipes[craft_idx].items():
            if k in ["_at", "_yield", "_step"]:
                continue

            num = inventory[k]
            inventory[k] -= min(v, num)
            if k in self.primitives:
                if num < v:
                    return_set.add(k)
            elif k in self.recipes:
                if num < v:
                    for _ in range(v - num):
                        return_set = self.find_missing_primitives(k, inventory, return_set)
            else:
                raise NotImplementedError
        return return_set

    def can_craft(self, craft_idx):
        return not self.find_missing_primitives(craft_idx)

    def possible_to_craft(self, craft_idx):
        for k, v in self.recipes[craft_idx].items():
            if k in ["_at", "_yield", "_step"]:
                if k == "_at" and v == self.furnace_idx and not self.env.state["furnace_ready"]:
                    return False
            elif k in self.primitive_idxes:
                tool_idx = self.primitives[k].get("_require", None)
                if tool_idx and not self.has_craft(tool_idx):
                    return False
            else:
                assert k in self.recipes, "Unknow key {}".format(k)
                if not self.possible_to_craft(k):
                    return False
        return True

    def has_craft(self, craft_idx):
        return self.env.has_craft(craft_idx)

    def in_furnace(self, craft_idx):
        return self.env.state["furnace_slot"] == self.cookbook.idx2furnace_slot[craft_idx]

    def has_treasure(self):
        return self.env.check_success()

    def can_collect_treasure(self):
        return self.env.can_collect_treasure()

    def select_craft(self, craft_idx):
        for k, v in self.recipes[craft_idx].items():
            if k in self.recipes:
                if self.env.inventory[k] < v:
                    return self.select_craft(k)
        return craft_idx

    def act(self, obs):
        env = self.env
        craft_goal = self.craft_goal
        collect_goal = self.collect_goal

        if np.random.rand() < self.random_action_prob:
            return np.random.randint(env.action_dim)

        self.update_state_machine()
        # print("state: {}, craft goal: {}, collect goal: {}".format(self.state, self.cookbook.index[self.craft_goal], self.collect_goal))

        random_craft = np.random.rand() < self.random_craft_prob
        grid = env.grid
        state = env.state

        if self.state == "COLLECT":
            if collect_goal is None:
                action = np.random.randint(env.action_dim)
            elif state[collect_goal + "_faced"]:
                action = USE
            else:
                action = env.move_actions[collect_goal]
        elif self.state == "CRAFT":
            station_idx = self.recipes[self.craft_goal]["_at"]
            if station_idx == self.workshop_idx:
                if state["workshop_faced"]:
                    if random_craft:
                        action = np.random.randint(len(self.craft_idxes))
                    else:
                        action = env.craft_idxes.index(self.craft_goal)
                    action += env.craft_action_starts
                else:
                    action = env.move_actions["workshop"]
            elif station_idx == self.furnace_idx:
                assert state["furnace_ready"]
                if self.in_furnace(self.craft_goal) and state["furnace_stage"] < self.recipes[self.craft_goal]["_step"]:
                    action = np.random.randint(env.action_dim)
                elif state["furnace_faced"]:
                    if self.in_furnace(self.craft_goal):
                        assert state["furnace_stage"] == self.recipes[self.craft_goal]["_step"]
                        action = USE
                    else:
                        if random_craft:
                            action = np.random.randint(len(self.craft_idxes))
                        else:
                            action = env.craft_idxes.index(self.craft_goal)
                        action += env.craft_action_starts
                else:
                    action = env.move_actions["furnace"]
            else:
                raise NotImplementedError
        elif self.state == "SETUP_FURNACE":
            assert env.inventory[self.furnace_idx]
            if state["furnace_faced"]:
                action = USE
            else:
                action = env.move_actions["furnace"]
        elif self.state == "COLLECT_TREASURE":
            has_path = any([not grid[x, y].any() for x, y, _ in env.neighbors(env.state[self.goal])])

            if state[self.goal + "_faced"]:
                action = USE
            elif has_path:
                action = env.move_actions[self.goal]
            else:
                obst_idx = self.primitives[self.goal_idx]["_surround"]
                obst_name = self.cookbook.index[obst_idx]
                num_obsts = int(len([k for k in state if obst_name in k]) / 3)
                obst_faced = any([state[obst_name + str(i) + "_faced"] for i in range(num_obsts)])
                if obst_faced:
                    action = USE
                else:
                    action = env.move_actions[obst_name + str(np.random.randint(num_obsts))]
        else:
            raise NotImplementedError

        self.num_collect_steps += 1

        return action
