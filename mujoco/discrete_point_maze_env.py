from .maze_env import MazeEnv
from .point import PointEnv
from .point_maze_env import PointMazeEnv
import gym
import math
import numpy as np


class DiscretePointMazeEnv(PointMazeEnv):
    def __init__(
            self,
            maze_id=None,
            maze_height=0.5,
            maze_size_scaling=8,
            n_bins=0,
            sensor_range=3.,
            sensor_span=2 * math.pi,
            observe_blocks=False,
            put_spin_near_agent=False,
            top_down_view=False,
            manual_collision=False,
            goal=None,
            disturb=False,

            amplify=1,

            *args,
            **kwargs):
        super(DiscretePointMazeEnv, self).__init__(maze_id,
                                                   maze_height,
                                                   maze_size_scaling,
                                                   n_bins,
                                                   sensor_range,
                                                   sensor_span,
                                                   observe_blocks,
                                                   put_spin_near_agent,
                                                   top_down_view,
                                                   manual_collision,
                                                   goal,



                                                   *args,
                                                   **kwargs)

        self.actual_action_space = self.action_space
        self.pseudo_action_space = gym.spaces.Discrete(4)
        self.disturb = False

        self.amplify = amplify


    def reset(self):
        obs = super(DiscretePointMazeEnv, self).reset()
        # try:
        #     obs = obs[0]['observation']
        # except IndexError:
        #     obs = obs
        # try:
        #     obs2 = obs['observation']
        # except IndexError:
        #     obs2 = obs
        # print("in reset", obs2)
        return obs

    def step(self, action):
        lb, ub = self.actual_action_space.low, self.actual_action_space.high
        disturb = np.random.randn(1) if self.disturb else 0
        if type(action) is not int:
            actual_action = np.array([0, 0])
        elif action == 0:

            actual_action = np.array([0, ub[1] * (1+0.1*disturb)* self.amplify])
        elif action == 1:
            actual_action = np.array([0, lb[1] * (1+0.1*disturb)* self.amplify])
        elif action == 2:
            actual_action = np.array([ub[0] * 0.5*(1+0.1*disturb)* self.amplify, 0])
        elif action == 3:
            actual_action = np.array([lb[0] * 0.5*(1+0.1*disturb)* self.amplify, 0])

        else:
            actual_action = np.array([0, 0])

        obs, reward, done, info = super(DiscretePointMazeEnv, self).step(actual_action)

        # print("in step", obs)
        # print("in step", obs2)
        return obs, reward, done, info
