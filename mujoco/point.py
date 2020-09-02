"""Wrapper for creating the ant environment in gym_mujoco."""

import math
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os


# class PointEnv(mujoco_env.MujocoEnv, utils.EzPickle):
#     FILE = "point.xml"
#     ORI_IND = 2
#
#     def __init__(self,
#                  target=None,
#                  wiggly_weight=0.,
#                  alt_xml=False,
#                  expose_velocity=True,
#                  expose_goal=False,
#                  use_simulator=False,
#                  file_path='point.xml'):
#         self._sample_target = target
#         if self._sample_target is not None:
#             self.goal = np.array([1.0, 1.0])
#         else:
#             self.goal = None
#         self._expose_velocity = expose_velocity
#         self._expose_goal = expose_goal
#         self._use_simulator = use_simulator
#         self._wiggly_weight = abs(wiggly_weight)
#         self._wiggle_direction = +1 if wiggly_weight > 0. else -1
#
#         # xml_path = "envs/assets/"
#         # model_path = os.path.abspath(os.path.join(xml_path, file_path))
#
#         if self._use_simulator:
#             mujoco_env.MujocoEnv.__init__(self, file_path, 5)
#         else:
#             mujoco_env.MujocoEnv.__init__(self, file_path, 1)
#         utils.EzPickle.__init__(self)
#
#     def step(self, action):
#         if self._use_simulator:
#             self.do_simulation(action, self.frame_skip)
#         else:
#             force = 0.2 * action[0]
#             rot = 1.0 * action[1]
#             qpos = self.sim.data.qpos.flat.copy()
#             qpos[2] += rot
#             ori = qpos[2]
#             dx = math.cos(ori) * force
#             dy = math.sin(ori) * force
#             qpos[0] = np.clip(qpos[0] + dx, -2, 2)
#             qpos[1] = np.clip(qpos[1] + dy, -2, 2)
#             qvel = self.sim.data.qvel.flat.copy()
#             self.set_state(qpos, qvel)
#
#         ob = self._get_obs()
#         if self._sample_target is not None and self.goal is not None:
#             reward = -np.linalg.norm(self.sim.data.qpos.flat[:2] - self.goal) ** 2
#         else:
#             reward = 0.
#
#         if self._wiggly_weight > 0.:
#             reward = (np.exp(-((-reward) ** 0.5)) ** (1. - self._wiggly_weight)) * (
#                     max(self._wiggle_direction * action[1], 0) ** self._wiggly_weight)
#         done = False
#         info = {}
#         return ob, reward, done, info
#
#     def _get_obs(self):
#         new_obs = [self.sim.data.qpos.flat]
#         if self._expose_velocity:
#             new_obs += [self.sim.data.qvel.flat]
#         if self._expose_goal and self.goal is not None:
#             new_obs += [self.goal]
#         return np.concatenate(new_obs)
#
#     def reset_model(self):
#         qpos = self.init_qpos+ self.np_random.uniform(
#             size=self.model.nq, low=-.1, high=.1)
#         qvel = self.init_qvel + self.np_random.randn(self.sim.model.nv) * .01
#         if self._sample_target is not None:
#             self.goal = self._sample_target(qpos[:2])
#         self.set_state(qpos, qvel)
#         return self._get_obs()
#
#     # only works when goal is not exposed
#     def set_qpos(self, state):
#         qvel = np.copy(self.sim.data.qvel.flat)
#         self.set_state(state, qvel)
#
#     def viewer_setup(self):
#         self.viewer.cam.distance = self.model.stat.extent * 0.5
#
#     def get_ori(self):
#             return self.data.qpos[self.__class__.ORI_IND]
#
#     def get_xy(self):
#         qpos = np.copy(self.data.qpos)
#         return qpos[:2]
#
#     def set_xy(self, xy):
#         qpos = np.copy(self.data.qpos)
#         qpos[0] = xy[0]
#         qpos[1] = xy[1]
#
#         qvel = self.data.qvel
#         self.set_state(qpos, qvel)
#
#     @property
#     def physics(self):
#         return self.model
class PointEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    FILE = "point.xml"
    ORI_IND = 2

    def __init__(self, file_path=None, expose_all_qpos=True):
        self._expose_all_qpos = expose_all_qpos

        mujoco_env.MujocoEnv.__init__(self, file_path, 1)
        utils.EzPickle.__init__(self)

    @property
    def physics(self):
        return self.model

    def _step(self, a):
        return self.step(a)

    def step(self, action):
        action[0] = 0.2 * action[0]
        qpos = np.copy(self.data.qpos)
        qpos[2] += action[1]
        ori = qpos[2]
        # compute increment in each direction
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]
        # ensure that the robot is within reasonable range
        qpos[0] = np.clip(qpos[0] + dx, -100, 100)
        qpos[1] = np.clip(qpos[1] + dy, -100, 100)
        qvel = self.data.qvel
        self.set_state(qpos, qvel)
        for _ in range(0, self.frame_skip):
            self.sim.step()
        next_obs = self._get_obs()
        reward = 0
        done = False
        info = {}
        return next_obs, reward, done, info

    def _get_obs(self):
        if self._expose_all_qpos:
            return np.concatenate([
                self.data.qpos.flat[:3],    # Only point-relevant coords.
                self.data.qvel.flat[:3]])
        return np.concatenate([
            self.data.qpos.flat[2:3],
            self.data.qvel.flat[:3]])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1

        # Set everything other than point to original position and 0 velocity.
        qpos[3:] = self.init_qpos[3:]
        qvel[3:] = 0.
        self.set_state(qpos, qvel)
        return self._get_obs()

    def get_ori(self):
        return self.data.qpos[self.__class__.ORI_IND]

    def set_xy(self, xy):
        qpos = np.copy(self.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]

        qvel = self.data.qvel
        self.set_state(qpos, qvel)

    def get_xy(self):
        qpos = np.copy(self.data.qpos)
        return qpos[:2]
