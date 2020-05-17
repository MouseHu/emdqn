# from rllab.envs.base import Step
# from .mujoco_env import MujocoEnv
# from rllab.core.serializable import Serializable
# from rllab.misc.overrides import overrides
import numpy as np
import math
# from rllab.mujoco_py import glfw

import os

from gym import spaces
from gym import utils
from gym.envs.mujoco import mujoco_env


class PointEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Point Environment"""
    def __init__(self):
        # super(PointEnv, self).__init__(*args, **kwargs)
        # Serializable.quick_init(self, locals())
        xml_file = 'point.xml'
        xml_file_path = os.path.join(os.path.dirname(__file__), '../assets', xml_file)
        mujoco_env.MujocoEnv.__init__(self, xml_file_path, 5)
        self.init_qacc = self.sim.data.qacc.ravel().copy()
        self.init_ctrl = self.sim.data.ctrl.ravel().copy()
        # self.current_com = None
        # self.dcom = None

        utils.EzPickle.__init__(self)

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy()
        low, high = bounds.T
        self.actual_action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        return self.action_space

    def step(self, action):
        lb, ub = self.actual_action_space.bounded_below, self.actual_action_space.bounded_above
        if action == 0:
            actual_action = np.array([0, ub[0] * 0.3])
        elif action == 1:
            actual_action = np.array([0, lb[0] * 0.3])
        elif action == 2:
            actual_action = np.array([ub[1], 0])
        elif action == 3:
            actual_action = np.array([lb[1], 0])
        else:
            actual_action = np.array([0, 0])

        qpos = np.copy(self.sim.data.qpos)
        qpos[2] += actual_action[1]
        ori = qpos[2]
        # compute increment in each direction
        dx = math.cos(ori) * actual_action[0]
        dy = math.sin(ori) * actual_action[0]
        # ensure that the robot is within reasonable range
        qpos[0] = np.clip(qpos[0] + dx, -7, 7)
        qpos[1] = np.clip(qpos[1] + dy, -7, 7)
        self.sim.data.qpos[:] = qpos
        self.sim.forward()
        obs = self._get_obs()

        return obs, 0, False, {}

    def _get_obs(self):
        data = self.sim.data
        cdists = np.copy(self.sim.model.geom_margin).flat
        for c in self.sim.data.contact:
            cdists[c.geom2] = min(cdists[c.geom2], c.dist)
        obs = np.concatenate([
            data.qpos.flat,
            data.qvel.flat,
            # data.cdof.flat,
            data.cinert.flat,
            data.cvel.flat,
            # data.cacc.flat,
            data.qfrc_actuator.flat,
            data.cfrc_ext.flat,
            data.qfrc_constraint.flat,
            cdists,
            # data.qfrc_bias.flat,
            # data.qfrc_passive.flat,
            # self.dcom.flat,
        ])

        return obs

    def reset_model(self):
        # qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.sim.nq)
        # qvel = self.init_qvel + self.np_random.randn(self.sim.nv) * .1
        # self.set_state(qpos, qvel)

        self.sim.data.qpos[:] = self.init_qpos + np.random.normal(size=self.init_qpos.shape) * 0.01
        self.sim.data.qvel[:] = self.init_qvel + np.random.normal(size=self.init_qvel.shape) * 0.1
        self.sim.data.qacc[:] = self.init_qacc
        self.sim.data.ctrl[:] = self.init_ctrl

        self.sim.forward()
        # self.current_com = self.sim.data.com_subtree[0]
        # self.dcom = np.zeros_like(self.current_com)

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.sim.model.stat.extent * 0.5

    # def step(self, action):
    #     qpos = np.copy(self.sim.data.qpos)
    #     qpos[2, 0] += action[1]
    #     ori = qpos[2, 0]
    #     # compute increment in each direction
    #     dx = math.cos(ori) * action[0]
    #     dy = math.sin(ori) * action[0]
    #     # ensure that the robot is within reasonable range
    #     qpos[0, 0] = np.clip(qpos[0, 0] + dx, -7, 7)
    #     qpos[1, 0] = np.clip(qpos[1, 0] + dy, -7, 7)
    #     self.sim.data.qpos = qpos
    #     self.sim.forward()
    #     next_obs = self.get_current_obs()
    #     return Step(next_obs, 0, False)
    #
    # def get_xy(self):
    #     qpos = self.sim.data.qpos
    #     return qpos[0, 0], qpos[1, 0]
    #
    # def set_xy(self, xy):
    #     qpos = np.copy(self.sim.data.qpos)
    #     qpos[0, 0] = xy[0]
    #     qpos[1, 0] = xy[1]
    #     self.sim.data.qpos = qpos
    #     self.sim.forward()

    # @overrides
    # def action_from_key(self, key):
    #     lb, ub = self.action_bounds
    #     if key == glfw.KEY_LEFT:
    #         return np.array([0, ub[0]*0.3])
    #     elif key == glfw.KEY_RIGHT:
    #         return np.array([0, lb[0]*0.3])
    #     elif key == glfw.KEY_UP:
    #         return np.array([ub[1], 0])
    #     elif key == glfw.KEY_DOWN:
    #         return np.array([lb[1], 0])
    #     else:
    #         return np.array([0, 0])
