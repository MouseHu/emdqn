import numpy as np
import os
import gym
import time
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from ple import PLE
from ple.games.monsterkong import MonsterKong


class MonsterKongEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, map_config, seed=1):
        self.map_config = map_config
        self.game = MonsterKong(self.map_config)

        self.fps = 30
        self.frame_skip = map_config['frame_skip']
        self.num_steps = 1
        self.force_fps = True
        self.display_screen = True
        self.nb_frames = 500
        self.reward = 0.0
        self.episode_end_sleep = 0.2

        if 'fps' in map_config:
            self.fps = map_config['fps']
        if 'frame_skip' in map_config:
            self.frame_skip = map_config['frame_skip']
        if 'force_fps' in map_config:
            self.force_fps = map_config['force_fps']
        if 'display_screen' in map_config:
            self.display_screen = map_config['display_screen']
        if 'episode_length' in map_config:
            self.nb_frames = map_config['episode_length']
        if 'episode_end_sleep' in map_config:
            self.episode_end_sleep = map_config['episode_end_sleep']
        self.current_step = 0

        self._seed(seed)

        self.p = PLE(self.game, fps=self.fps, frame_skip=self.frame_skip, num_steps=self.num_steps,
                     force_fps=self.force_fps, display_screen=self.display_screen, rng=self.rng)

        self.p.init()

        self._action_set = self.p.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        (screen_width, screen_height) = self.p.getScreenDims()
        self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3))

    def _seed(self, seed=24):

        self.rng = seed
        if seed != 24:
            np.random.seed(seed)

    def step(self, action_taken):
        reward = 0.0
        # print(self._action_set, action_taken)
        action = np.array(self._action_set)[action_taken]

        reward += self.p.act(action)
        # print(reward)
        obs = self.p.getScreenRGB()
        done = self.p.game_over()
        self.current_step += 1
        if done:
            info = [{'PLE': self.p, 'episode': {'r': reward, 'l': self.current_step}}]
        else:
            info = [{'PLE': self.p}]
        if self.current_step >= self.nb_frames:
            done = True
        return obs, reward, done, info

    def reset(self):
        self.current_step = 0
        # Noop and reset if done
        start_done = True
        while start_done:
            self.p.reset_game()
            _, _, start_done, _ = self.step(5)
            # self.p.init()
        if self.p.display_screen:
            self.render()
            if self.episode_end_sleep > 0:
                time.sleep(self.episode_end_sleep)
        return self.p.getScreenRGB()

    def render(self, mode='human', close=False):
        if close:
            return  # TODO: implement close
        original = self.p.display_screen
        self.p.display_screen = True
        self.p._draw_frame()
        self.p.display_screen = original
