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


class LRU_KNN_KBPS(object):
    def __init__(self, num_actions, buffer_size, latent_dim, hash_dim, gamma=0.99, bp=True, debug=True):
        self.num_actions = num_actions
        self.gamma = gamma
        self.rmax = 100000
        self.bp = bp
        self.debug = debug
        self.ec_buffer = LRU_KNN_GPU_PS(buffer_size, latent_dim, hash_dim, 'game', num_actions, debug=debug)
        self.logger = logging.getLogger("ecbp")
        self.pqueue = HashPQueue()
        self.sa_explore = 10
        self.max_iter = 1000
        self.dist = None
        self.ind = None
        self.b = 10

    # def act_value(self, keys, action, knn):
    #     return self.ec_buffer.act_value(keys, knn)

    def log(self, *args, logtype='debug', sep=' '):
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    def grow_model(self, sa_pair):  # grow model
        index_t, action_t, reward_t, z_tp1, done_t = sa_pair
        index_tp1 = self.peek(z_tp1)

        if index_tp1 < 0:
            index_tp1, override = self.ec_buffer.add_node(z_tp1)
            self.log("add node", index_tp1, logtype='debug')
            if override:
                self.pqueue.remove(index_tp1)

        # if (index_t, action_t) not in self.ec_buffer.prev_id[index_tp1]:
        self.log("add edge", index_t, action_t, index_tp1, logtype='debug')
        sa_count = self.ec_buffer.add_edge(index_t, index_tp1, action_t, reward_t, done_t)
        coeff = np.exp(self.dist / self.b)
        self.ec_buffer.pseudo_count[index_t, action_t] = {}
        for i, s in enumerate(self.ind):
            for sp in self.ec_buffer.next_id[s][action_t].keys():
                try:
                    self.ec_buffer.pseudo_count[index_t, action_t] += coeff[i] * self.ec_buffer.next_id[s][action_t][sp]
                except KeyError:
                    self.ec_buffer.pseudo_count[index_t, action_t] = coeff[i] * self.ec_buffer.next_id[s][action_t][sp]

            self.ec_buffer.pseudo_reward[index_t, action_t] += coeff[i] * self.ec_buffer.reward[s, action_t]
            for sp in self.ec_buffer.next_id[index_t][action_t].keys():
                try:
                    self.ec_buffer.pseudo_count[s, action_t] += coeff[i] * self.ec_buffer.next_id[index_tp1][action_t][
                        sp]
                except KeyError:
                    self.ec_buffer.pseudo_count[s, action_t] = coeff[i] * self.ec_buffer.next_id[index_tp1][action_t][
                        sp]
            self.ec_buffer.pseudo_reward[s, action_t] += coeff[i] * self.ec_buffer.reward[index_t, action_t]
        if sa_count > self.sa_explore:
            self.ec_buffer.internal_value[index_t, action_t] = 0
        return index_tp1, sa_count

    def prioritized_sweeping(self, sa_pair):
        # grow model
        index_tp1, count_t = self.grow_model(sa_pair)
        # update current value
        index_t, action_t, reward_t, z_tp1, done_t = sa_pair
        assert index_t in self.ind, "self should be a neighbor of self"
        for index in self.ind:
            self.update_q_value(index, action_t)
            self.ec_buffer.state_value_v[index_t] = max(self.ec_buffer.external_value[index_t, :])
            priority = abs(self.ec_buffer.state_value_v[index_t] - self.ec_buffer.state_value_u[index_t])
            if priority > 1e-7:
                self.pqueue.push(priority, index_t)
        # recursive backup
        self.log("begin backup")
        num_iters = 0
        while len(self.pqueue) > 0 and num_iters < self.max_iter:
            num_iters += 1
            priority, state = self.pqueue.pop()
            delta_u = self.ec_buffer.state_value_v[state] - self.ec_buffer.state_value_u[state]
            self.ec_buffer.state_value_u[state] = self.ec_buffer.state_value_v[state]
            self.log("backup node", state, "priority", priority, "new value", self.ec_buffer.state_value_v[state],
                     "delta", delta_u)
            for sa_pair in self.ec_buffer.prev_id[state]:
                state_tm1, action_tm1 = sa_pair
                self.log("update s,a,s',delta", state_tm1, action_tm1, state, delta_u)
                self.update_q_value_backup(state_tm1, action_tm1, state, delta_u)
                self.ec_buffer.state_value_v[state_tm1] = max(self.ec_buffer.external_value[state_tm1, :])
                priority = abs(self.ec_buffer.state_value_v[state_tm1] - self.ec_buffer.state_value_u[state_tm1])
                if priority > 1e-4:
                    self.pqueue.push(priority, state_tm1)
        self.log("finish backup")
        return index_tp1

    def peek(self, state):
        ind = self.ec_buffer.peek(state)
        return ind

    def update_q_value(self, state, action):

        n_sa = sum(self.ec_buffer.pseudo_count[state][action].values())
        r_smooth = self.ec_buffer.pseudo_reward[state, action] / n_sa
        # n_sasp = sum([coeff[i] * self.ec_buffer.next_id[s][action].get(state_tp1, 0) for i, s in enumerate(self.ind)])
        self.ec_buffer.external_value[state, action] = r_smooth
        for state_tp1 in self.ec_buffer.pseudo_count.keys():
            value_tp1 = self.ec_buffer.state_value_u[state_tp1]
            trans_p = self.ec_buffer.pseudo_count[state][action][state_tp1] / n_sa
            self.ec_buffer.external_value[state, action] += trans_p * self.gamma * value_tp1

    def update_q_value_backup(self, state, action, state_tp1, delta_u):
        n_sa = sum(self.ec_buffer.pseudo_count[state][action].values())
        n_sasp = self.ec_buffer.pseudo_count[state][action].get(state_tp1, 0)
        trans_p = n_sasp / n_sa
        self.ec_buffer.external_value[state, action] += self.gamma * trans_p * delta_u
