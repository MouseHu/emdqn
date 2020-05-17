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
import threading
from multiprocessing import Process
from multiprocessing import Lock, Event
from multiprocessing import Manager


class BackUpProcess(Process):
    def __init__(self, ns, lock):
        super(BackUpProcess, self).__init__()
        self.ns = ns
        self.pqueue = ns.pqueue
        self.run_sweep = True
        self.iters_per_step = 0
        self.max_iter = 10000
        self.num_iters = 0
        self.logger = logging.getLogger("ecbp")
        self.queue_lock = lock
        self.ec_buffer = ns.ec_buffer

    def log(self, *args, logtype='debug', sep=' '):
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    def run(self):
        self.backup()

    def backup(self):
        # recursive backup
        # self.log("begin backup")
        while self.run_sweep:
            self.num_iters += 1
            self.log("bk pqueue len", len(self.pqueue))
            if len(self.pqueue) > 0:
                self.log("what is wrong?")
                self.queue_lock.acquire()
                priority, index = self.pqueue.pop()
                self.queue_lock.release()
                delta_u = self.ec_buffer.state_value_v[index] - np.nan_to_num(self.ec_buffer.state_value_u[index],
                                                                              copy=True)
                self.ec_buffer.state_value_u[index] = self.ec_buffer.state_value_v[index]
                self.log("backup node", index, "priority", priority, "new value",
                         self.ec_buffer.state_value_v[index],
                         "delta", delta_u)
                for sa_pair in self.ec_buffer.prev_id[index]:
                    state_tm1, action_tm1 = sa_pair
                    # self.log("update s,a,s',delta", state_tm1, action_tm1, index, delta_u)
                    self.ec_buffer.update_q_value(state_tm1, action_tm1, index, delta_u)
                    self.ec_buffer.state_value_v[state_tm1] = np.nanmax(self.ec_buffer.external_value[state_tm1, :])
                    priority = abs(
                        self.ec_buffer.state_value_v[state_tm1] - np.nan_to_num(
                            self.ec_buffer.state_value_u[state_tm1], copy=True))
                    if priority > 1e-7:
                        self.queue_lock.acquire()
                        self.pqueue.push(priority, state_tm1)
                        self.queue_lock.release()
            if self.num_iters % 100000 == 0:
                self.log("backup count", self.num_iters)
            # if (self.iters_per_step >= self.max_iter or len(self.pqueue) == 0) and not self.update_enough.is_set():
            #     self.update_enough.set()


class LRU_KNN_PRIORITIZEDSWEEPING(object):
    def __init__(self, num_actions, buffer_size, latent_dim, hash_dim, gamma=0.99, bp=True, debug=True):
        self.num_actions = num_actions
        self.gamma = gamma
        self.rmax = 100000
        self.bp = bp
        self.debug = debug
        self.manager = Manager()
        self.ns = self.manager.Namespace()
        self.logger = logging.getLogger("ecbp")
        self.sa_explore = 10
        self.max_iter = 1000000
        self.run_sweep = True
        self.num_iters = 0
        # self.queue_lock = Lock()
        self.ns.ec_buffer = LRU_KNN_GPU_PS(buffer_size, latent_dim, hash_dim, 'game', num_actions, debug=debug)
        self.ec_buffer = self.ns.ec_buffer
        self.ns.pqueue = HashPQueue()
        self.pqueue = self.ns.pqueue
        self.queue_lock = self.manager.Lock()
        self.sweep_process = BackUpProcess(self.ns, self.queue_lock)
        self.begin_sweep()

    # def act_value(self, keys, action, knn):
    #     return self.ec_buffer.act_value(keys, knn)

    def begin_sweep(self):
        # self.sweep_process = Process(target=LRU_KNN_PRIORITIZEDSWEEPING.backup, args=(self, self.ns, self.queue_lock))
        self.sweep_process.run_sweep = True
        self.sweep_process.start()

    def end_sweep(self):
        # self.sweep_process.run_sweep = False
        self.sweep_process.join()

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

    def prioritized_sweeping(self, sa_pair):
        # self.update_enough.wait(timeout=1000)
        self.log("ps pqueue len", len(self.pqueue))
        # grow model
        index_tp1, count_t = self.grow_model(sa_pair)
        # update current value
        index_t, action_t, reward_t, z_tp1, done_t = sa_pair
        value_tp1 = np.nan_to_num(self.ec_buffer.state_value_u[index_tp1], copy=True)
        # value_tp1 = 0
        self.log("ps update", index_t, action_t, count_t, reward_t + self.gamma * value_tp1,
                 self.ec_buffer.external_value[index_t, action_t])
        # self.ec_buffer.external_value[index_t, action_t] = reward_t + self.gamma * value_tp1 * (1 - done_t)
        if np.isnan(self.ec_buffer.external_value[index_t, action_t]):
            self.ec_buffer.external_value[index_t, action_t] = 0
        self.ec_buffer.external_value[index_t, action_t] += 1 / count_t * (
                reward_t + self.gamma * value_tp1 - self.ec_buffer.external_value[index_t, action_t])
        self.log("u,v pre", self.ec_buffer.state_value_v[index_t], self.ec_buffer.state_value_u[index_t])
        self.log("after ps update", self.ec_buffer.external_value[index_t, :])
        self.ec_buffer.state_value_v[index_t] = np.nanmax(self.ec_buffer.external_value[index_t, :])
        self.log("u,v post", self.ec_buffer.state_value_v[index_t], self.ec_buffer.state_value_u[index_t])
        priority = abs(
            self.ec_buffer.state_value_v[index_t] - np.nan_to_num(self.ec_buffer.state_value_u[index_t], copy=True))
        if priority > 1e-7:
            self.queue_lock.acquire()
            self.pqueue.push(priority, index_t)
            self.queue_lock.release()
            self.log("add queue", priority, len(self.pqueue))
        # self.iters_per_step = 0
        # self.update_enough.clear()
        return index_tp1

    def backup(self, ns, lock):
        # recursive backup
        self.log("begin backup", self.run_sweep)
        ec_buffer = ns.ec_buffer
        pqueue = ns.pqueue
        while self.run_sweep:
            self.num_iters += 1
            self.log("bk pqueue len", len(pqueue))
            if len(pqueue) > 0:
                self.log("what is wrong?")
                lock.acquire()
                priority, index = pqueue.pop()
                lock.release()
                delta_u = ec_buffer.state_value_v[index] - np.nan_to_num(ec_buffer.state_value_u[index],
                                                                         copy=True)
                ec_buffer.state_value_u[index] = ec_buffer.state_value_v[index]
                self.log("backup node", index, "priority", priority, "new value",
                         ec_buffer.state_value_v[index],
                         "delta", delta_u)
                for sa_pair in ec_buffer.prev_id[index]:
                    state_tm1, action_tm1 = sa_pair
                    # self.log("update s,a,s',delta", state_tm1, action_tm1, index, delta_u)
                    ec_buffer.update_q_value(state_tm1, action_tm1, index, delta_u)
                    ec_buffer.state_value_v[state_tm1] = np.nanmax(ec_buffer.external_value[state_tm1, :])
                    priority = abs(
                        ec_buffer.state_value_v[state_tm1] - np.nan_to_num(
                            ec_buffer.state_value_u[state_tm1], copy=True))
                    if priority > 1e-7:
                        lock.acquire()
                        pqueue.push(priority, state_tm1)
                        lock.release()
            if self.num_iters % 100000 == 0:
                self.log("backup count", self.num_iters)
        self.log("finish backup", self.run_sweep)

    def peek(self, state):
        ind = self.ec_buffer.peek(state)
        return ind
