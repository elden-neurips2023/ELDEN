import random
import numpy as np


class ScriptedPhysical:
    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.action_dim = params.action_dim

        self.num_objects = env.num_objects
        self.num_rand_objects = env.num_rand_objects
        self.width = env.width
        self.height = env.height
        self.directions = [Coord(-1, 0),
                           Coord(0, 1),
                           Coord(1, 0),
                           Coord(0, -1)]
        self.policy_id = 0
        self.reset()

    def reset(self, *args):
        policy_id = self.policy_id
        self.mov_obj_idx = policy_id // (self.num_objects + self.num_rand_objects - 1)
        self.target_obj_idx = policy_id % (self.num_objects + self.num_rand_objects - 1)
        if self.target_obj_idx >= self.mov_obj_idx:
            self.target_obj_idx += 1
        self.direction_idx = np.random.randint(4)
        self.direction = self.directions[self.direction_idx]
        self.success_steps = 0
        self.random_policy = np.random.rand() < 0.1

        n_policies = self.num_objects * (self.num_objects + self.num_rand_objects - 1)
        self.policy_id = (policy_id + 1) % n_policies

    def get_action(self, obj_idx, offset):
        if obj_idx >= self.num_objects:
            return 5 * np.random.randint(self.num_objects)
        return 5 * obj_idx + self.directions.index(offset) + 1

    def dijkstra(self, obj_idx_to_move, target_pos):
        env = self.env
        width, height = env.width, env.height
        Q = np.ones((width, height)) * np.inf
        dist = np.ones((width, height)) * np.inf
        checked = np.zeros((width, height), dtype=bool)
        for idx, obj in env.objects.items():
            checked[obj.pos.x, obj.pos.y] = True

        Q[target_pos.x, target_pos.y] = 0

        while True:
            x, y = np.unravel_index(np.argmin(Q), Q.shape)
            q = Q[x, y]
            if q == np.inf:
                break
            dist[x, y] = Q[x, y]
            checked[x, y] = True
            Q[x, y] = np.inf

            for del_x, del_y in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                new_x, new_y = x + del_x, y + del_y
                if 0 <= new_x < width and 0 <= new_y < height and not checked[new_x, new_y]:
                    if q + 1 < Q[new_x, new_y]:
                        Q[new_x, new_y] = q + 1

        mov_obj = env.objects[obj_idx_to_move]
        mov_x, mov_y = mov_obj.pos.x, mov_obj.pos.y
        min_dist = np.inf
        min_idx = self.directions[0]
        for dir in self.directions:
            new_x, new_y = mov_x + dir.x, mov_y + dir.y
            if 0 <= new_x < width and 0 <= new_y < height:
                if dist[new_x, new_y] < min_dist:
                    min_dist = dist[new_x, new_y]
                    min_idx = dir
        return min_idx, min_dist

    def act(self, obs):
        objects = self.env.objects
        env = self.env
        mov_obj_idx = self.mov_obj_idx
        target_obj_idx = self.target_obj_idx
        mov_obj = objects[mov_obj_idx]
        target_obj = objects[target_obj_idx]

        current_pos = mov_obj.pos
        target_pos = target_obj.pos - self.direction

        map_center = Coord(self.width // 2 + 1, self.height // 2 + 1)
        # need to push the target object from outside of the map (impossible), need to adjust the target object
        if not 0 <= target_pos.x < self.width or not 0 <= target_pos.y < self.height:
            if env.valid_move(target_obj_idx, self.direction):
                return self.get_action(target_obj_idx, self.direction)
            else:
                action_idx, dist = self.dijkstra(target_obj_idx, map_center)
                return self.get_action(target_obj_idx, action_idx)

        # unable to simply move the target object, need to plan a path for it instead (by letting it move to the center)
        pushing_pos = target_obj.pos + self.direction
        if any([obj.pos == pushing_pos for obj in objects.values() if obj != mov_obj]):
            action_idx, dist = self.dijkstra(target_obj_idx, map_center)
            if dist != np.inf:
                return self.get_action(target_obj_idx, action_idx)

        if current_pos != target_pos:
            if self.random_policy:
                return self.get_action(mov_obj_idx, random.choice(self.directions))
            else:
                action_idx, dist = self.dijkstra(mov_obj_idx, target_pos)
                if dist == np.inf:
                    return self.get_action(target_obj_idx, self.direction)
                else:
                    return self.get_action(mov_obj_idx, action_idx)
        else:
            action = self.get_action(mov_obj_idx, self.direction)
            self.success_steps += 1
            if self.success_steps > 2:
                self.reset()
            return action
