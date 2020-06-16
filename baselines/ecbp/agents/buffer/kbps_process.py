import numpy as np
from sklearn.neighbors import BallTree, KDTree
import os
from baselines.ecbp.agents.buffer.lru_knn_gpu_ps import LRU_KNN_GPU_PS
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


class KernelBasedPriorSweepProcess(Process):
    def __init__(self, num_actions, buffer_size, latent_dim, hash_dim, conn, gamma=0.99):
        super(KernelBasedPriorSweepProcess, self).__init__()
        self.num_actions = num_actions
        self.gamma = gamma
        self.rmax = 100000
        self.logger = logging.getLogger("ecbp")
        self.sa_explore = 10
        self.max_iter = 1000000
        self.run_sweep = True
        self.num_iters = 0
        self.conn = conn
        self.buffer_size = buffer_size
        self.latent_dim = latent_dim
        self.hash_dim = hash_dim
        # self.queue_lock = Lock()
        self.pqueue = HashPQueue()
        self.b = 0.01
        self.h = 0.01
        self.knn_dist = None
        self.knn_ind = None
        self.sequence = []

    def log(self, *args, logtype='debug', sep=' '):
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    def grow_model(self, sa_pair):  # grow model
        index_t, action_t, reward_t, z_tp1, done_t = sa_pair
        index_tp1, _, _ = self.peek(z_tp1)

        if index_tp1 < 0:
            index_tp1, override = self.ec_buffer.add_node(z_tp1)
            self.log("add node", index_tp1, logtype='debug')
            if override:
                self.pqueue.remove(index_tp1)

        # if (index_t, action_t) not in self.ec_buffer.prev_id[index_tp1]:
        self.log("add edge", index_t, action_t, index_tp1, logtype='debug')
        sa_count = self.ec_buffer.add_edge(index_t, index_tp1, action_t, reward_t, done_t)
        coeff = np.exp(-np.array(self.knn_dist).reshape(-1) / self.b)
        self.log("coeff", coeff.shape, coeff)
        self.ec_buffer.pseudo_count[index_t][action_t] = {}
        self.ec_buffer.pseudo_reward[index_t, action_t] = 0
        # self.ec_buffer.pseudo_prev[index_tp1] = {}
        assert index_t in self.knn_ind, "self should be a neighbour of self"
        for i, s in enumerate(self.knn_ind):

            for sp in self.ec_buffer.next_id[s][action_t].keys():
                dist = self.ec_buffer.distance(self.ec_buffer.states[sp],
                                               self.ec_buffer.states[sp] + self.ec_buffer.states[index_t] -
                                               self.ec_buffer.states[s])
                reweight = np.exp(-np.array(dist).squeeze() / self.h)
                weighted_count = reweight * coeff[i] * self.ec_buffer.next_id[s][action_t][sp]
                try:
                    self.ec_buffer.pseudo_count[index_t][action_t][sp] += weighted_count
                except KeyError:
                    self.ec_buffer.pseudo_count[index_t][action_t][sp] = weighted_count

                self.ec_buffer.pseudo_prev[sp][(index_t, action_t)] = 1
                self.ec_buffer.pseudo_reward[index_t, action_t] += reweight * coeff[i] * self.ec_buffer.reward[
                    s, action_t]
            if index_t == s:
                continue
            for sp in self.ec_buffer.next_id[index_t][action_t].keys():
                dist = self.ec_buffer.distance(self.ec_buffer.states[sp],
                                               self.ec_buffer.states[sp] + self.ec_buffer.states[s] -
                                               self.ec_buffer.states[index_t])
                reweight = np.exp(-np.array(dist).squeeze() / self.h)
                weighted_count = reweight * coeff[i] * self.ec_buffer.next_id[index_t][action_t][sp]
                try:
                    self.ec_buffer.pseudo_count[s][action_t][sp] += weighted_count
                except KeyError:
                    self.ec_buffer.pseudo_count[s][action_t][sp] = weighted_count
                self.ec_buffer.pseudo_prev[sp][(s, action_t)] = 1
                self.ec_buffer.pseudo_reward[s, action_t] += reweight * coeff[i] * self.ec_buffer.reward[
                    index_t, action_t]
        if sa_count > self.sa_explore:
            self.ec_buffer.internal_value[index_t, action_t] = 0
        return index_tp1, sa_count

    def observe(self, sa_pair):
        # self.update_enough.wait(timeout=1000)
        # self.log("ps pqueue len", len(self.pqueue))
        # grow model
        index_tp1, count_t = self.grow_model(sa_pair)
        # update current value
        index_t, action_t, reward_t, z_tp1, done_t = sa_pair
        self.sequence.append(index_t)
        self.log("self neighbour", index_t, self.knn_ind)
        assert index_t in self.knn_ind, "self should be a neighbor of self"
        for index in self.knn_ind:
            # self.log("q before observe", self.ec_buffer.external_value[index, :],index,action_t)
            self.update_q_value(index, action_t)
            # self.log("q after observe", self.ec_buffer.external_value[index, :], index, action_t)
            self.ec_buffer.state_value_v[index_t] = np.nanmax(self.ec_buffer.external_value[index_t, :])
            priority = abs(
                self.ec_buffer.state_value_v[index_t] - np.nan_to_num(self.ec_buffer.state_value_u[index_t], copy=True))
            if priority > 1e-7:
                self.pqueue.push(priority, index_t)
        if done_t:
            self.update_sequence()
        # self.iters_per_step = 0
        # self.update_enough.clear()
        self.conn.send((2, index_tp1))

    def backup(self):
        # recursive backup
        self.num_iters += 1
        if len(self.pqueue) > 0:
            priority, index = self.pqueue.pop()
            delta_u = self.ec_buffer.state_value_v[index] - np.nan_to_num(self.ec_buffer.state_value_u[index],
                                                                          copy=True)
            self.ec_buffer.state_value_u[index] = self.ec_buffer.state_value_v[index]
            self.log("backup node", index, "priority", priority, "new value",
                     self.ec_buffer.state_value_v[index],
                     "delta", delta_u)
            for sa_pair in self.ec_buffer.pseudo_prev[index].keys():
                state_tm1, action_tm1 = sa_pair
                # self.log("update s,a,s',delta", state_tm1, action_tm1, index, delta_u)
                # self.log("q before backup",self.ec_buffer.external_value[state_tm1,:],state_tm1,action_tm1)
                self.update_q_value_backup(state_tm1, action_tm1, index, delta_u)
                self.ec_buffer.state_value_v[state_tm1] = np.nanmax(self.ec_buffer.external_value[state_tm1, :])
                # self.log("q after backup", self.ec_buffer.external_value[index, :], state_tm1,action_tm1)
                priority = abs(
                    self.ec_buffer.state_value_v[state_tm1] - np.nan_to_num(
                        self.ec_buffer.state_value_u[state_tm1], copy=True))
                if priority > 1e-7:
                    self.pqueue.push(priority, state_tm1)
        if self.num_iters % 100000 == 0:
            self.log("backup count", self.num_iters)

    def update_sequence(self):
        # to make sure that the final signal can be fast propagate through the state,
        # we need a sequence update like episodic control
        for p, s in enumerate(self.sequence):
            # self.pqueue.push(p + self.rmax, s)
            self.ec_buffer.newly_added[s] = False
        self.sequence = []

        # self.ec_buffer.build_tree()

    def update_q_value(self, state, action):

        n_sa = sum(self.ec_buffer.pseudo_count[state][action].values())
        if n_sa < 1e-7:
            return
        r_smooth = np.nan_to_num(self.ec_buffer.pseudo_reward[state, action] / n_sa)
        # n_sasp = sum([coeff[i] * self.ec_buffer.next_id[s][action].get(state_tp1, 0) for i, s in enumerate(self.ind)])
        self.ec_buffer.external_value[state, action] = r_smooth
        for state_tp1 in self.ec_buffer.pseudo_count[state][action].keys():
            value_tp1 = np.nan_to_num(self.ec_buffer.state_value_u[state_tp1])
            trans_p = self.ec_buffer.pseudo_count[state][action][state_tp1] / n_sa
            self.ec_buffer.external_value[state, action] += trans_p * self.gamma * value_tp1

    def update_q_value_backup(self, state, action, state_tp1, delta_u):
        n_sa = sum(self.ec_buffer.pseudo_count[state][action].values())
        if n_sa < 1e-7:
            return
        n_sasp = self.ec_buffer.pseudo_count[state][action].get(state_tp1, 0)
        trans_p = n_sasp / n_sa
        assert 0 <= trans_p <= 1, "nsa{} nsap{} trans{}".format(n_sa, n_sasp, trans_p)
        if np.isnan(self.ec_buffer.external_value[state, action]):
            self.ec_buffer.external_value[state, action] = 0
        self.ec_buffer.external_value[state, action] += self.gamma * trans_p * delta_u

    def peek(self, state):
        ind = self.ec_buffer.peek(state)
        return ind

    def run(self):
        self.ec_buffer = LRU_KNN_GPU_PS(self.buffer_size, self.hash_dim, 'game', 0, self.num_actions)
        while self.run_sweep:
            self.backup()
            self.recv_msg()

    def retrieve_q_value(self, obj):
        z, knn = obj
        extrinsic_qs, intrinsic_qs, find = self.ec_buffer.act_value_ec(z, knn)
        self.conn.send((0, (extrinsic_qs, intrinsic_qs, find)))

    def peek_node(self, obj):
        z = obj
        ind, knn_dist, knn_ind = self.ec_buffer.peek(z)
        knn_dist = np.array(knn_dist).reshape(-1).tolist()
        knn_ind = np.array(knn_ind).reshape(-1).tolist()
        if ind == -1:
            ind, _ = self.ec_buffer.add_node(z)
            knn_dist = [0] + knn_dist
            knn_ind = [ind] + knn_ind
            self.log("add node for first ob ", ind)
        self.knn_dist = knn_dist
        self.knn_ind = knn_ind
        self.conn.send((1, ind))

    def recv_msg(self):
        # 0 —— retrieve q values
        # 1 —— peek or add node
        # 2 —— observe
        # 3 —— kill
        while self.conn.poll():
            msg, obj = self.conn.recv()
            if msg == 0:
                self.retrieve_q_value(obj)
            elif msg == 1:
                self.peek_node(obj)
            elif msg == 2:
                self.observe(obj)
            elif msg == 3:
                self.run_sweep = False
                self.conn.send((3, True))
            else:
                raise NotImplementedError
