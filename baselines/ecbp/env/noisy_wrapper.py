import gym
import numpy as np
import time


class StaticNoisyEnv(gym.Wrapper):
    def __init__(self, env, noisy_dim, noisy_var, noisy_count=1324573, seed=None):
        super(StaticNoisyEnv, self).__init__(env)
        assert noisy_dim > 0 and noisy_var > 0
        assert noisy_count > 0
        assert type(env.observation_space) is gym.spaces.Box
        if seed is None:
            self.seed = int(time.time())
        else:
            self.seed = seed
        np.random.seed(self.seed)
        self.noisy_dim = noisy_dim
        self.noisy_var = noisy_var
        self.noisy_count = noisy_count
        self.noise = np.random.randn(noisy_count, noisy_dim)
        low = np.concatenate((-4 * noisy_var * np.ones(noisy_dim), np.array(env.observation_space.low)))
        high = np.concatenate((-4 * noisy_var * np.ones(noisy_dim), np.array(env.observation_space.low)))
        self.observation_space = gym.spaces.Box(low, high)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        noise = self.noise[hash(tuple(obs)) % self.noisy_count]
        new_obs = np.concatenate((noise, obs))
        return new_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        noise = self.noise[hash(tuple(obs)) % self.noisy_count]
        new_obs = np.concatenate((noise, obs))
        return new_obs


class DynamicNoisyEnv(gym.Wrapper):
    def __init__(self, env, noisy_dim, noisy_var, noisy_count=13, seed=None):
        super(DynamicNoisyEnv, self).__init__(env)
        assert noisy_dim > 0 and noisy_var > 0
        assert noisy_count > 0
        assert type(env.observation_space) is gym.spaces.Box
        if seed is None:
            self.seed = int(time.time())
        else:
            self.seed = seed
        np.random.seed(self.seed)
        self.noisy_dim = noisy_dim
        self.noisy_var = noisy_var
        self.noisy_count = noisy_count
        self.noise = np.random.randn(noisy_count, noisy_dim)
        low = np.concatenate((-4 * noisy_var * np.ones(noisy_dim), np.array(env.observation_space.low)))
        high = np.concatenate((-4 * noisy_var * np.ones(noisy_dim), np.array(env.observation_space.low)))
        self.observation_space = gym.spaces.Box(low, high)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        noise = self.noise[np.random.randint(0,self.noisy_count)]
        new_obs = np.concatenate((noise, obs))
        return new_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        noise = self.noise[np.random.randint(0,self.noisy_count)]
        new_obs = np.concatenate((noise, obs))
        return new_obs


class DiscreteWrapper(gym.Wrapper):
    def __init__(self, env, **kwargs):
        super(DiscreteWrapper, self).__init__(env)
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
        self.action_space = gym.spaces.Discrete(9)  # up, up-right, right, down-right, down, down-left,
        print(self.observation_space.spaces)
        self.observation_space = self.observation_space.spaces['image']

    def reset(self):
        obs = self.env.reset()
        return obs["image"]

    def step(self, action):

        if type(action) is not int or action == 8:  # no-op
            cont_action = self.disc2cont_actions[8]
        elif 0 <= action <= 8:
            cont_action = self.disc2cont_actions[action]
        else:
            raise ValueError("Unknown action: {}".format(action))

        cont_action = self._scalar * cont_action

        obs, reward, done, info = self.env.step(cont_action)
        return obs["image"], reward, done, info


class AdditiveStaticNoisyEnv(gym.Wrapper):
    def __init__(self, env, noisy_var, noisy_count=13):
        super(AdditiveStaticNoisyEnv, self).__init__(env)
        assert noisy_var >= 0
        assert type(env.observation_space) is gym.spaces.Box
        self.noisy_var = noisy_var
        self.noisy_count = noisy_count
        low = env.observation_space.low - 10 * noisy_var * np.ones_like(env.observation_space.shape)
        high = env.observation_space.high + 10 * noisy_var * np.ones_like(env.observation_space.shape)
        self.observation_space = gym.spaces.Box(low, high)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        noise = self.noisy_var * np.random.randn(self.noisy_dim)
        new_obs = np.concatenate((noise, obs))
        return new_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        noise = self.noisy_var * np.random.randn(self.noisy_dim)
        new_obs = np.concatenate((noise, obs))
        return new_obs
