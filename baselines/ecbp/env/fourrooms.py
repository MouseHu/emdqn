import numpy as np
import gym
import time
from gym import error, spaces
from gym import core, spaces
from gym.envs.registration import register
import random

# resultList = random.sample(range(15, 300 + 15), 104)


class Fourrooms(object):
    # metadata = {'render.modes':['human']}
    def __init__(self):
        layout = """\
1111111111111
1     1     1
1     1     1
1           1
1     1     1
1     1     1
11 1111     1
1     111 111
1     1     1
1     1     1
1           1
1     1     1
1111111111111
"""
        self.occupancy = np.array([list(map(lambda c: 1 if c == '1' else 0, line)) for line in layout.splitlines()])

        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))

        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        # self.rng = np.random.RandomState(1234)

        self.tostate = {}
        self.semantics = dict()
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i, j)] = statenum
                    statenum += 1
        self.tocell = {v: k for k, v in self.tostate.items()}

        self.goal = 62

        self.init_states = list(range(self.observation_space.n))
        self.init_states.remove(self.goal)
        # random encode
        self.mapping = np.arange(int(np.sum(self.occupancy == 0)))
        self.dict = np.zeros((self.observation_space.n, 3))
        self.Row = np.shape(self.occupancy)[0]
        self.Col = np.shape(self.occupancy)[1]

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        # state = self.rng.choice(self.init_states)
        state = np.random.choice(self.init_states)
        self.currentcell = self.tocell[state]
        return self.mapping[state]

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.

        We consider a case in which rewards are zero on all state transitions.
        """
        nextcell = tuple(self.currentcell + self.directions[action])

        if not self.occupancy[nextcell]:
            self.currentcell = nextcell
            if np.random.uniform() < 0.:
                # if self.rng.uniform() < 1/3.:
                empty_cells = self.empty_around(self.currentcell)
                # self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]
                self.currentcell = empty_cells[np.random.randint(len(empty_cells))]

        state = self.tostate[self.currentcell]
        done = state == self.goal

        # print(self.currentcell)
        return self.mapping[state], done, None

    def get_dict(self):
        count = 0
        for i in range(self.Row):
            for j in range(self.Col):
                if self.occupancy[i, j] == 0:
                    # code
                    self.dict[count, 0] = self.mapping[count]
                    # i,j
                    self.dict[count, 1] = i
                    self.dict[count, 2] = j

                    self.semantics[self.mapping[count]] = str(i) + '_' + str(j)
                    count += 1

        # print(self.semantics)
        return self.semantics

# register(
#     id='Fourrooms-v0',
#     entry_point='fourrooms:Fourrooms',
#     timestep_limit=20000,
#     reward_threshold=1,
# )
