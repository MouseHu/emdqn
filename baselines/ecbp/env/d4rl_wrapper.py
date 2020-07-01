import gym
import numpy as np
from d4rl.offline_env import OfflineEnv


class D4RLDiscreteMazeEnvWrapper(gym.Wrapper, OfflineEnv):
    def __init__(self, env, **kwargs):
        gym.Wrapper.__init__(self, env)
        OfflineEnv.__init__(self, dataset_url=env.dataset_url, ref_max_score=env.ref_max_score,
                            ref_min_score=env.ref_min_score, **kwargs)

        self.pseudo_action_space = gym.spaces.Discrete(9)  # up, up-right, right, down-right, down, down-left,
                                                           # left, up-left, no-op
        self.disc2cont_actions = {
            0: np.array([0.0, 1.0]),
            1: np.array([1.0, 1.0]),
            2: np.array([1.0, 0.0]),
            3: np.array([1.0, -1.0]),
            4: np.array([0.0, -1.0]),
            5: np.array([-1.0, -1.0]),
            6: np.array([-1.0, 0.0]),
            7: np.array([-1.0, 1.0]),
            8: np.array([0.0, 0.0])
        }
        self._scalar = 0.5  # constant

    def step(self, action):

        if type(action) is not int or action == 8:  # no-op
            cont_action = self.disc2cont_actions[8]
        elif 0 <= action <= 8:
            cont_action = self.disc2cont_actions[action]
        else:
            raise ValueError("Unknown action: {}".format(action))

        cont_action = self._scalar * cont_action

        return self.env.step(cont_action)

    def get_dataset(self, h5path=None):
        dataset = super().get_dataset(h5path)
        # discretize actions
        dataset['raw_actions'] = dataset.pop('actions')

        actions = []
        round_cont_actions = np.round(dataset['raw_actions'])
        cont_actions = np.array(list(self.disc2cont_actions.values()))
        for round_cont_action in round_cont_actions:
            action = int(np.argwhere(np.all(cont_actions == round_cont_action, axis=1))[0][0])
            actions.append(action)
        dataset['actions'] = np.array(actions, dtype=np.int32)
        dataset['next_observations'] = np.concatenate([dataset['observations'][1:], dataset['observations'][-1][None]],
                                                      axis=0)

        return dataset
