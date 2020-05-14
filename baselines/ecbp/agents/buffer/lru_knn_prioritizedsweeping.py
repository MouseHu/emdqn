import numpy as np
from sklearn.neighbors import BallTree, KDTree
import os
from baselines.ecbp.agents.buffer.lru_knn_gpu_ps import LRU_KNN_GPU_PS
import gc
from baselines.deepq.experiments.atari.knn_cuda_fixmem import knn as knn_cuda_fixmem
import copy
from heapq import *
import logging
from baselines.ecbp.agents.buffer.hash_pqueue import HashPQueue


class LRU_KNN_PRIORITIZEDSWEEPING(object):
    def __init__(self, num_actions, buffer_size, latent_dim, hash_dim, gamma=0.99, bp=True, debug=True):
        self.num_actions = num_actions
        self.gamma = gamma
        self.rmax = 100000
        self.bp = bp
        self.debug = debug
        self.ec_buffer = LRU_KNN_GPU_PS(buffer_size, latent_dim, hash_dim, 'game', num_actions, debug=debug)
        self.logger = logging.getLogger("ecbp")
        self.pqueue = HashPQueue()
        self.sa_explore = 1
        self.max_iter = 1000

    # def act_value(self, keys, action, knn):
    #     return self.ec_buffer.act_value(keys, knn)

    def log(self, *args, logtype='debug', sep=' '):
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    def grow_model(self, sa_pair):  # grow model
        index_t, action_t, reward_t, z_tp1, done_t = sa_pair
        index_tp1 = self.peek(z_tp1)

        if index_tp1 < 0:
            index_tp1 = self.ec_buffer.add_node(z_tp1)
            self.log("add node", index_tp1, logtype='debug')

        # if (index_t, action_t) not in self.ec_buffer.prev_id[index_tp1]:
        self.log("add edge", index_t, action_t, index_tp1, logtype='debug')
        sa_count = self.ec_buffer.add_edge(index_t, index_tp1, action_t, reward_t, done_t)
        # if sa_count > self.sa_explore:
        #     self.ec_buffer.internal_value[index_t, action_t] = 0
        return index_tp1, sa_count

    def prioritized_sweeping(self, sa_pair):
        # grow model
        index_tp1, count_t = self.grow_model(sa_pair)
        # update current value
        index_t, action_t, reward_t, z_tp1, done_t = sa_pair
        value_tp1 = self.ec_buffer.state_value_u[index_tp1]
        self.log("ps update", index_t, count_t, reward_t + self.gamma * value_tp1,
                 self.ec_buffer.external_value[index_t, action_t])
        # self.ec_buffer.external_value[index_t, action_t] = reward_t + self.gamma * value_tp1 * (1 - done_t)

        self.ec_buffer.external_value[index_t, action_t] += 1 / count_t * (
                reward_t + self.gamma * value_tp1 - self.ec_buffer.external_value[index_t, action_t])
        self.ec_buffer.state_value_v[index_t] = max(self.ec_buffer.external_value[index_t, :])
        priority = abs(self.ec_buffer.state_value_v[index_t] - self.ec_buffer.state_value_u[index_t])
        if priority > 1e-7:
            self.pqueue.push(priority, index_t)
        # recursive backup
        self.log("begin backup")
        num_iters = 0
        while len(self.pqueue) > 0 and num_iters < self.max_iter:
            num_iters += 1
            priority, index = self.pqueue.pop()
            delta_u = self.ec_buffer.state_value_v[index] - self.ec_buffer.state_value_u[index]
            self.ec_buffer.state_value_u[index] = self.ec_buffer.state_value_v[index]
            self.log("backup node", index, "priority", priority, "new value", self.ec_buffer.state_value_v[index],
                     "delta", delta_u)
            for sa_pair in self.ec_buffer.prev_id[index]:
                stm1, atm1 = sa_pair
                self.log("update s,a,s',delta", stm1, atm1, index, delta_u)
                self.ec_buffer.update_q_value(stm1, atm1, index, delta_u)
                self.ec_buffer.state_value_v[stm1] = max(self.ec_buffer.external_value[stm1, :])
                priority = abs(self.ec_buffer.state_value_v[stm1] - self.ec_buffer.state_value_u[stm1])
                if priority > 1e-4:
                    self.pqueue.push(priority, stm1)
        self.log("finish backup")
        return index_tp1

    def peek(self, state):
        ind = self.ec_buffer.peek(state)
        return ind
