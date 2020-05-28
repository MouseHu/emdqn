import numpy as np
from sklearn.neighbors import BallTree, KDTree
import os
from baselines.ecbp.agents.buffer.lru_knn_gpu_ps import LRU_KNN_GPU_PS
from baselines.ecbp.agents.buffer.lru_knn_count_gpu_fixmem import LRU_KNN_COUNT_GPU_FIXMEM
from baselines.ecbp.agents.buffer.ps_learning_process import PSLearningProcess
from baselines.ecbp.agents.buffer.lru_knn_ps import LRU_KNN_PS
import gc
from baselines.deepq.experiments.atari.knn_cuda_fixmem import knn as knn_cuda_fixmem
import copy
from heapq import *
import logging
from baselines.ecbp.agents.buffer.hash_pqueue import HashPQueue
import threading
from multiprocessing import Process
from multiprocessing import Lock, Event
from multiprocessing import Manager


class ECLearningProcess(PSLearningProcess):
    def __init__(self, num_actions, buffer_size, latent_dim, obs_dim, conn, gamma=0.99, knn=4, queue_threshold=5e-5):
        super(ECLearningProcess, self).__init__(num_actions, buffer_size, latent_dim, obs_dim, conn, gamma, knn,
                                                queue_threshold)

    def observe(self, sa_pair):
        # self.update_enough.wait(timeout=1000)
        # self.log("ps pqueue len", len(self.pqueue))
        # grow model

        # update current value

        z_t, index_t, action_t, reward_t, z_tp1, h_tp1, done_t = sa_pair
        # self.log("begin grow model")
        # index_tp1, count_t = self.grow_model((index_t, action_t, reward_t, z_tp1, h_tp1, done_t))
        # self.log("finish grow model")
        self.sequence.append((z_t, reward_t, action_t))
        if done_t:
            self.update_sequence()
            # self.iters_per_step = 0
        # self.update_enough.clear()
        # assert index_tp1 != -1
        self.conn.send((2, 0))

    def update_sequence(self):
        # to make sure that the final signal can be fast propagate through the state,
        # we need a sequence update like episodic control
        Rtn = 0
        for p, experience in enumerate(reversed(self.sequence)):
            z, r, a = experience
            Rtn = r + self.gamma * Rtn
            # self.ec_buffer.external_value[s,a] = max(Rtn,self.ec_buffer.external_value[s,a])
            q, _ = self.ec_buffer[a].peek(z, Rtn, 0, True)
            self.log("update sequence", q, Rtn, a)
            if q is None:  # new action
                self.ec_buffer[a].add(z, Rtn, self.rmax)
        self.sequence = []
        # self.ec_buffer.build_tree()

    def retrieve_q_value(self, obj):
        z, h, knn = obj
        # extrinsic_qs, intrinsic_qs, find = self.ec_buffer.act_value_ec(z, knn)
        # self.conn.send((0, (extrinsic_qs, intrinsic_qs, find)))
        # self.log("send finish")
        extrinsic_qs = np.zeros((self.num_actions, 1))
        intrinsic_qs = np.zeros((self.num_actions, 1))
        finds = np.zeros((self.num_actions,))
        for a in range(self.num_actions):
            extrinsic_qs[a], intrinsic_qs[a], find = self.ec_buffer[a].act_value(np.array([z]), knn)
            finds[a] = sum(find)
        self.log("capacity",[self.ec_buffer[a].curr_capacity for a in range(self.num_actions)])
        self.log("find", finds)
        self.conn.send((0, (extrinsic_qs, intrinsic_qs, finds)))

    def run(self):
        # self.ec_buffer = LRU_KNN_GPU_PS(self.buffer_size, self.latent_dim, 'game', 0, self.num_actions,
        #                                 self.knn)
        self.ec_buffer = [
            LRU_KNN_COUNT_GPU_FIXMEM(self.buffer_size, self.latent_dim, "game", a, self.num_actions, self.knn) for a in
            range(self.num_actions)]
        while self.run_sweep:
            self.recv_msg()
            # self.update_enough = 0
