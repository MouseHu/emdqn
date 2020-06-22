import gym
import numpy as np
from d4rl.offline_env import OfflineEnv


class D4RLDiscreteMazeEnvWrapper(gym.Wrapper, OfflineEnv):
    def __init__(self, env, **kwargs):
        gym.Wrapper.__init__(self, env)
        OfflineEnv.__init__(self, **kwargs)

        self.actual_action_space = self.env.action_space
        self.action_space = gym.spaces.Discrete(9)  # up, up-right, right, down-right, down, down-left,
                                                    # left, up-left, no-op

    def step(self, action):

        if type(action) is not int or action == 8:  # no-op
            action = np.array([0.0, 0.0])
        elif action == 0:  # up
            action = np.array([0.0, 1.0])
        elif action == 1:  # up-right
            action = np.array([1.0, 1.0])
        elif action == 2:  # right
            action = np.array([1.0, 0.0])
        elif action == 3:  # down-right
            action = np.array([1.0, -1.0])
        elif action == 4:  # down
            action = np.array([0.0, -1.0])
        elif action == 5:  # down-left
            action = np.array([-1.0, -1.0])
        elif action == 6:  # left
            action = np.array([-1.0, 0.0])
        elif action == 7:  # up-left
            action = np.array([-1.0, 1.0])
        else:
            raise ValueError("Unknown action: {}".format(action))

        scalar = 0.5  # constant
        action = scalar * action

        return self.env.step(action)
