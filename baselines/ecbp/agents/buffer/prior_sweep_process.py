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


class PriorSweepProcess(Process):
    def __init__(self, num_actions, buffer_size, latent_dim, hash_dim, conn, gamma=0.99, knn=4, queue_threshold=5e-5):
        super(PriorSweepProcess, self).__init__()
        self.num_actions = num_actions
        self.gamma = gamma
        self.rmax = 100000
        self.logger = logging.getLogger("ecbp")
        self.sa_explore = 10
        self.min_iter = 20
        self.run_sweep = True
        self.num_iters = 0
        self.conn = conn
        self.buffer_size = buffer_size
        self.latent_dim = latent_dim
        self.hash_dim = hash_dim
        self.knn = knn
        self.update_enough = True
        self.iters_per_step = 0
        self.queue_threshold = queue_threshold
        # self.queue_lock = Lock()
        self.pqueue = HashPQueue()
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

        # if sa_count > self.sa_explore:
        #     self.ec_buffer.internal_value[index_t, action_t] = 0
        return index_tp1, sa_count

    def observe(self, sa_pair):
        # self.update_enough.wait(timeout=1000)
        # self.log("ps pqueue len", len(self.pqueue))
        # grow model
        index_tp1, count_t = self.grow_model(sa_pair)
        # update current value
        index_t, action_t, reward_t, z_tp1, done_t = sa_pair
        self.sequence.append(index_t)
        if done_t:
            #     delayed update, so that the value can be efficiently propagated
            if np.isnan(self.ec_buffer.external_value[index_t, action_t]):
                self.ec_buffer.external_value[index_t, action_t] = reward_t
                total_count_tp1 = sum(self.ec_buffer.next_id[index_t][action_t].values())
                for s_tp1, count_tp1 in self.ec_buffer.next_id[index_t][action_t].items():
                    trans_p = count_tp1 / total_count_tp1
                    value_tp1 = np.nan_to_num(self.ec_buffer.state_value_u[s_tp1])
                    self.ec_buffer.external_value[index_t, action_t] += self.gamma * trans_p * value_tp1
            else:
                value_tp1 = np.nan_to_num(self.ec_buffer.state_value_u[index_tp1], copy=True)
                self.ec_buffer.external_value[index_t, action_t] += 1 / count_t * (
                        reward_t + self.gamma * value_tp1 - self.ec_buffer.external_value[index_t, action_t])
            self.ec_buffer.state_value_v[index_t] = np.nanmax(self.ec_buffer.external_value[index_t, :])
            self.update_sequence()
            self.conn.send((2, index_tp1))
            return

        if np.isnan(self.ec_buffer.external_value[index_t, action_t]) and np.isnan(self.ec_buffer.state_value_u[index_tp1]):
            # if next value is nan, we can't infer anything about current q value,
            # so we should return immediately witout update q values
            self.conn.send((2, index_tp1))
            return
        value_tp1 = np.nan_to_num(self.ec_buffer.state_value_u[index_tp1], copy=True)
        self.log("ps update", index_t, action_t, count_t, reward_t + self.gamma * value_tp1,
                 self.ec_buffer.external_value[index_t, action_t])
        if np.isnan(self.ec_buffer.external_value[index_t, action_t]):
            self.ec_buffer.external_value[index_t, action_t] = 0
        # self.ec_buffer.external_value[index_t, action_t] = reward_t + self.gamma * value_tp1 * (1 - done_t)
        self.ec_buffer.external_value[index_t, action_t] += 1 / count_t * (
                reward_t + self.gamma * value_tp1 - self.ec_buffer.external_value[index_t, action_t])
        self.log("u,v pre", self.ec_buffer.state_value_v[index_t], self.ec_buffer.state_value_u[index_t])
        self.log("after ps update", self.ec_buffer.external_value[index_t, :])
        self.ec_buffer.state_value_v[index_t] = np.nanmax(self.ec_buffer.external_value[index_t, :])
        self.log("u,v post", self.ec_buffer.state_value_v[index_t], self.ec_buffer.state_value_u[index_t])
        priority = abs(
            self.ec_buffer.state_value_v[index_t] - np.nan_to_num(self.ec_buffer.state_value_u[index_t], copy=True))
        if priority > self.queue_threshold:
            self.pqueue.push(priority, index_t)
            self.log("add queue", priority, len(self.pqueue))
        # self.iters_per_step = 0
        # self.update_enough.clear()
        assert index_tp1 != -1
        self.conn.send((2, index_tp1))

    def update_sequence(self):
        # to make sure that the final signal can be fast propagate through the state,
        # we need a sequence update like episodic control
        for p, s in enumerate(self.sequence):
            self.pqueue.push(p + self.rmax, s)
            self.ec_buffer.newly_added[s] = False
        self.sequence = []
        self.update_enough = False
        self.iters_per_step = 0
        # self.ec_buffer.build_tree()

    def backup(self):
        # recursive backup
        # self.log("begin backup", self.run_sweep)
        self.num_iters += 1
        # self.log("bk pqueue len", len(self.pqueue))
        if len(self.pqueue) > 0:
            if self.iters_per_step < self.min_iter:
                self.iters_per_step += 1
            # self.log("what is wrong?")
            priority, index = self.pqueue.pop()
            delta_u = self.ec_buffer.state_value_v[index] - np.nan_to_num(self.ec_buffer.state_value_u[index],
                                                                          copy=True)
            self.ec_buffer.state_value_u[index] = self.ec_buffer.state_value_v[index]
            self.log("backup node", index, "priority", priority, "new value",
                     self.ec_buffer.state_value_v[index],
                     "delta", delta_u)
            # self.log("pqueue len",len(self.pqueue))
            for sa_pair in self.ec_buffer.prev_id[index]:
                state_tm1, action_tm1 = sa_pair
                # self.log("update s,a,s',delta", state_tm1, action_tm1, index, delta_u)
                self.ec_buffer.update_q_value(state_tm1, action_tm1, index, delta_u)
                self.ec_buffer.state_value_v[state_tm1] = np.nanmax(self.ec_buffer.external_value[state_tm1, :])
                priority_tm1 = abs(
                    self.ec_buffer.state_value_v[state_tm1] - np.nan_to_num(
                        self.ec_buffer.state_value_u[state_tm1], copy=True))
                if priority_tm1 > self.queue_threshold:
                    self.pqueue.push(priority_tm1, state_tm1)
            if priority < self.rmax and not self.update_enough:
                self.update_enough = True
                self.log("update enough with low priority", self.update_enough, priority)
        if len(self.pqueue) == 0 and not self.update_enough:
            self.update_enough = True
            self.log("update enough", self.update_enough)
        if self.num_iters % 100000 == 0:
            self.log("backup count", self.num_iters)

    def peek(self, state):
        ind = self.ec_buffer.peek(state)
        return ind

    def run(self):
        self.ec_buffer = LRU_KNN_GPU_PS(self.buffer_size, self.latent_dim, self.hash_dim, 'game', self.num_actions,
                                        self.knn)
        while self.run_sweep:
            self.backup()
            if self.update_enough:
                self.recv_msg()
                # self.update_enough = 0

    def retrieve_q_value(self, obj):
        z, knn = obj
        extrinsic_qs, intrinsic_qs, find = self.ec_buffer.act_value(z, knn)
        self.conn.send((0, (extrinsic_qs, intrinsic_qs, find)))

    def peek_node(self, obj):
        z = obj
        ind, _, _ = self.ec_buffer.peek(z)
        if ind == -1:
            ind, _ = self.ec_buffer.add_node(z)
            self.log("add node for first ob ", ind)
        assert ind != -1
        self.conn.send((1, ind))

    def recv_msg(self):
        # 0 —— retrieve q values
        # 1 —— peek or add node
        # 2 —— observe
        # 3 —— kill
        while self.conn.poll():
            # self.update_enough = False
            # self.iters_per_step = 0
            self.log("receiving message, update enough", self.update_enough)
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
