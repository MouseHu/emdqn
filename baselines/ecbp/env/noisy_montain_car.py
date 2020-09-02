import gym
import numpy as np


class NoisyEnv(gym.Wrapper):
    def __init__(self, env, noisy_dim, noisy_var):
        super(NoisyEnv, self).__init__(env)
        assert noisy_dim > 0 and noisy_var > 0
        assert type(env.observation_space) is gym.spaces.Box
        self.noisy_dim = noisy_dim
        self.noisy_var = noisy_var
        low = -4 * noisy_var * np.ones(noisy_dim) + env.observation_space.low
        high = -4 * noisy_var * np.ones(noisy_dim) + env.observation_space.low
        self.observation_space = gym.spaces.Box(low, high, (env.shape[0] + self.noisy_dim,))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        noise = self.noisy_var * np.random.randn(self.noisy_dim)
        new_obs = noise + obs
        return new_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        noise = self.noisy_var * np.random.randn(self.noisy_dim)
        new_obs = noise + obs
        return new_obs
