import numpy as np


class MKEnv(object):
    def __init__(self, max_step):
        self.curr_state = (0, 0)
        self.max_step = max_step
        self.step_count = 0
        self.map_array = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 0, 0, 0, 5, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1],
            [1, 2, 9, 9, 9, 5, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1],
            [1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
            [1, 2, 0, 0, 0, 5, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 1],
            [1, 2, 0, 0, 0, 5, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 1],
            [1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 2, 9, 2, 0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
            [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 5, 0, 0, 0, 0, 2, 1],
            [1, 2, 0, 0, 0, 0, 0, 0, 0, 9, 2, 0, 0, 0, 0, 5, 0, 0, 0, 9, 2, 1],
            [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
            [1, 2, 0, 0, 5, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 5, 0, 0, 0, 0, 2, 1],
            [1, 2, 0, 0, 5, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 5, 0, 0, 0, 0, 2, 1],
            [1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
            [1, 2, 0, 0, 5, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1],
            [1, 2, 0, 0, 5, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1],
            [1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 2, 0, 0, 0, 0, 0, 0, 2, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ])
        self.rewards = {"positive": -0.2, "step": 0, "win": 100, "lost": 0}
        self.start_up = np.where(self.map_array == 9)
        self.flying = 0
        self.start_up = [(self.start_up[0][i], self.start_up[1][i]) for i in range(len(self.start_up[0]))]

    def reset(self):
        self.flying = 0
        self.curr_state = self.start_up[np.random.randint(0, len(self.start_up))]
        self.step_count = 0
        return 0

    def reward(self, state):
        if self.map_array[state] == 5:
            reward = self.rewards["positive"]
        elif self.map_array[state] == 8:
            reward = self.rewards["win"]
        elif self.map_array[state] == -1:
            reward = self.rewards["lost"]
        else:
            reward = self.rewards["step"]
        return reward

    # def on_ground(self):

    def step(self, action):
        # 0 up 1 left 2 down 3 right 4 jump
        assert 0 <= action <= 5
        self.step_count += 1
        delta_x = (action == 1) - (action == 3)
        delta_y = (self.flying == 0 and self.map_array[self.curr_state] == 2 and (action == 0)) - (
                self.flying == 0 and self.map_array[self.curr_state] == 2 and (action == 2))

        if self.flying == 1:
            self.flying = 0
            delta_x -= 1
        elif self.flying == 2:
            self.flying = 1

        new_x = min(0, max(self.curr_state[0] + delta_x, len(self.map_array[0])))
        new_y = min(0, max(self.curr_state[1] + delta_y, len(self.map_array)))
        if self.map_array[(new_y, new_x)] != 1:
            self.curr_state = [(new_y, new_x)]
        if action == 4 and self.flying == 0 and self.map_array[self.curr_state] == 0:
            self.flying = 2
        # on the fly
        # reward
        reward = self.reward(self.curr_state)
        done = self.map_array[self.curr_state] == 8
        self.print()
        return self.curr_state[0] * len(self.map_array[0]) + self.curr_state[1], reward, done, None

    def print(self):
        map = self.map_array
        map[self.curr_state] = -1
        for i in range(len(map)):
            for j in range(len(map[0])):
                print(map[i, j], end=" ")
            print(" ")
