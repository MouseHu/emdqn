import numpy as np
import gym
import time
from gym import error, spaces
from gym import core, spaces
from gym.envs.registration import register
import random

from baselines.ecbp.env import rendering

# resultList = random.sample(range(15, 300 + 15), 104)


class Tworooms(gym.Env):
    # metadata = {'render.modes':['human']}
    def __init__(self):
        layout = """\
1111111111111
1           1
1           1
1           1
1           1
1           1
1222222222331
1           1
1           1
1           1
1           1
1           1
1111111111111
"""
        d = {'0':0,'1':1,'2':2,'3':3}
        self.occupancy = np.array([list(map(lambda c: d.get(c,0), line)) for line in layout.splitlines()])
        print(self.occupancy)
        self.row_num,self.col_num = self.occupancy.shape
        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.space_capacity = np.sum(self.occupancy == 0)
        self.observation_space = spaces.Box(np.zeros(self.space_capacity),np.ones(self.space_capacity))

        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # self.rng = np.random.RandomState(1234)

        self.tostate = {}
        self.semantics = dict()
        statenum = 0
        for i in range(self.row_num):
            for j in range(self.col_num):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i, j)] = statenum
                    statenum += 1
        self.tocell = {v: k for k, v in self.tostate.items()}

        self.goal = 0

        self.init_states = list(range(int(self.space_capacity)))
        self.init_states.remove(self.goal)
        # random encode
        self.num_state = 0
        self.cell_state  = self.tocell[self.num_state]
        self.max_step = 100
        self.num_steps = 0

        self.block_size = 20

        self.viewer = rendering.Viewer(self.block_size * len(self.occupancy), self.block_size * len(self.occupancy[0]))

        self.blocks = self.make_blocks()

    def set_state(self,s):
        self.num_state = s
        self.cell_state = self.tocell[self.num_state]

    def reset(self):
        self.num_state = np.random.choice(self.init_states)
        self.cell_state = self.tocell[self.num_state]
        self.num_steps = 0
        return self.one_hot(self.num_state)

    def one_hot(self,x):
        assert 0<= x < self.space_capacity
        state = np.zeros(self.space_capacity)
        state[x] = 1
        return state

    def step(self, action):
        assert 0 <= action < 4
        self.num_steps+=1
        self.move(action)
        reward = 0
        if self.occupancy[self.cell_state] == 2:
            reward = -10
        if self.occupancy[self.cell_state] >= 2:
            self.move(action)

        self.num_state = self.tostate[self.cell_state]
        if self.num_state == self.goal:
            reward= 1
        done = (self.num_steps>=self.max_step) or (self.num_state == self.goal)
        return self.one_hot(self.num_state),reward,done,None

    def move(self,action):
        x = self.cell_state[0]+self.directions[action][0]
        y = self.cell_state[1]+self.directions[action][1]
        if self.occupancy[x][y] ==1:
            return
        else:
            self.cell_state = (x,y)

    def add_block(self, x, y, color):
        size = self.block_size
        v = [[x * size, y * size], [x * size, (y + 1) * size], [(x + 1) * size, (y + 1) * size],
             [(x + 1) * size, y * size]]

        geom = rendering.make_polygon(v, filled=True)

        r, g, b = color
        geom.set_color(r, g, b)
        self.viewer.add_onetime(geom)

    def make_blocks(self):
        blocks = []
        size = self.block_size
        for i, row in enumerate(self.occupancy):
            for j, o in enumerate(row):
                if o == 1 or o == 2:
                    v = [[j * size, i * size], [j * size, (i + 1) * size], [(j + 1) * size, (i + 1) * size],
                         [(j + 1) * size, (i) * size]]

                    geom = rendering.make_polygon(v, filled=True)

                    if o == 1:
                        geom.set_color(0, 0, 0)
                    elif o == 2:
                        geom.set_color(0, 0, 0)
                    blocks.append(geom)
                    self.viewer.add_geom(geom)
        return blocks

    def render(self,mode=0):
        if self.cell_state[0] > 0:
            x, y = self.cell_state
            self.add_block(x, y, (0, 0, 1))

        x, y = self.tocell[self.goal]
        self.add_block(x, y, (1, 0, 0))
        # self.viewer.
        self.viewer.render(return_rgb_array=True)
        return
