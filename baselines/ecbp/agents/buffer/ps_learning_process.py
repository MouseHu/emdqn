import numpy as np
from sklearn.neighbors import BallTree, KDTree
import os
from baselines.ecbp.agents.buffer.lru_knn_gpu_ps import LRU_KNN_GPU_PS
from baselines.ecbp.agents.buffer.lru_knn_gpu_ps_density import LRU_KNN_GPU_PS_DENSITY
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

import pickle as pkl


class PSLearningProcess(Process):
    def __init__(self, num_actions, buffer_size, latent_dim, obs_dim, conn, gamma=0.99, knn=4, queue_threshold=5e-5,
                 density=True):

        super(PSLearningProcess, self).__init__()
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
        self.obs_dim = obs_dim
        self.knn = knn
        self.update_enough = True
        self.iters_per_step = 0
        self.queue_threshold = queue_threshold
        # self.queue_lock = Lock()
        self.pqueue = HashPQueue()
        self.sequence = []

        self.first_ob_index = -10
        self.use_density = density
        self.ec_buffer = None


    def log(self, *args, logtype='debug', sep=' '):
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    def grow_model(self, sa_pair):  # grow model
        index_t, action_t, reward_t, z_tp1, h_tp1, done_t = sa_pair
        index_tp1, knn_dist, knn_ind = self.ec_buffer.peek(z_tp1)
        # self.log("finish peek")
        if index_tp1 < 0:

            index_tp1, override = self.ec_buffer.add_node(z_tp1, knn_dist, knn_ind)
            # index_tp1, override = self.ec_buffer.add_node(z_tp1)


            # self.log("add node", index_tp1, logtype='debug')
            if override:
                self.pqueue.remove(index_tp1)

        # if (index_t, action_t) not in self.ec_buffer.prev_id[index_tp1]:
        # self.log("add edge", index_t, action_t, index_tp1, logtype='debug')
        sa_count = self.ec_buffer.add_edge(index_t, index_tp1, action_t, reward_t, done_t)

        # if sa_coun t > self.sa_explore:
        #     self.ec_buffer.internal_value[index_t, action_t] = 0
        return index_tp1, sa_count


    def save(self, filedir):
        while len(self.pqueue) > 0:  # empty pqueue
            self.backup()
        ec_buffer_file = open(os.path.join(filedir, "ec_buffer.pkl"), "wb")
        pkl.dump(self.ec_buffer, ec_buffer_file)

    def load(self, filedir):
        try:
            ec_buffer_file = open(os.path.join(filedir, "ec_buffer.pkl"), "rb")
            self.ec_buffer = pkl.load(ec_buffer_file)
        except FileNotFoundError:
            return
        self.ec_buffer.allocate()
        batch_size = 32
        for i in range(int(np.ceil((self.buffer_size + 1) / batch_size))):
            low = i * batch_size
            high = min(self.buffer_size, (i + 1) * batch_size)
            z_to_update = self.ec_buffer.states[low:high]

            # print(z_to_update.shape,np.arange(low, high))
            # self.log("z shape", np.array(z_to_update).shape)
            self.ec_buffer.update(np.arange(low, high), np.array(z_to_update))


    def observe(self, sa_pair):
        # self.update_enough.wait(timeout=1000)
        # self.log("ps pqueue len", len(self.pqueue))
        # grow model
        index_tp1, count_t = self.grow_model(sa_pair)
        # update current value
        index_t, action_t, reward_t, z_tp1, h_tp1, done_t = sa_pair
        self.sequence.append(index_t)
        if done_t:
            #     delayed update, so that the value can be efficiently propagated
            if np.isnan(self.ec_buffer.external_value[index_t, action_t]):
                # self.ec_buffer.notify(index_t, action_t)
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

        if np.isnan(self.ec_buffer.external_value[index_t, action_t]) and np.isnan(
                self.ec_buffer.state_value_u[index_tp1]):
            # if next value is nan, we can't infer anything about current q value,
            # so we should return immediately witout update q values
            self.conn.send((2, index_tp1))
            return
        value_tp1 = np.nan_to_num(self.ec_buffer.state_value_u[index_tp1], copy=True)

        self.log("ps update s,a,count,new_value,old_value", index_t, action_t, count_t,
                 reward_t + self.gamma * value_tp1,
                 self.ec_buffer.external_value[index_t, action_t])
        if np.isnan(self.ec_buffer.external_value[index_t, action_t]):
            self.ec_buffer.external_value[index_t, action_t] = 0
        self.ec_buffer.external_value[index_t, action_t] += 1 / count_t * (
                reward_t + self.gamma * value_tp1 - self.ec_buffer.external_value[index_t, action_t])
        self.ec_buffer.state_value_v[index_t] = np.nanmax(self.ec_buffer.external_value[index_t, :])

        priority = abs(
            self.ec_buffer.state_value_v[index_t] - np.nan_to_num(self.ec_buffer.state_value_u[index_t], copy=True))
        if priority > self.queue_threshold:
            self.pqueue.push(priority, index_t)

            # self.log("add queue", priority, len(self.pqueue))
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
                # self.log("update enough with low priority", self.update_enough, priority)
        if len(self.pqueue) == 0 and not self.update_enough:
            self.update_enough = True
            # self.log("update enough", self.update_enough)
        if self.num_iters % 100000 == 0:
            self.log("backup count", self.num_iters)

    # def peek(self, state):
    #     ind = self.ec_buffer.peek(state)
    #     return ind

    def run(self):

        buffer = LRU_KNN_GPU_PS_DENSITY if self.use_density else LRU_KNN_GPU_PS
        self.ec_buffer = buffer(self.buffer_size, self.latent_dim, 'game', 0, self.num_actions,
                                self.knn)

        while self.run_sweep:
            self.backup()
            if self.update_enough:
                self.recv_msg()
                # self.update_enough = 0

    def retrieve_q_value(self, obj):
        z, h, knn = obj

        extrinsic_qs, intrinsic_qs, find, neighbour_ind = self.ec_buffer.act_value(z, knn)
        self.conn.send((0, (extrinsic_qs, intrinsic_qs, find,neighbour_ind)))


    def peek_node(self, obj):
        z, h = obj
        ind, knn_dist, knn_ind = self.ec_buffer.peek(z)
        if ind == -1:

            ind, _ = self.ec_buffer.add_node(z, knn_dist, knn_ind)
            # ind, _ = self.ec_buffer.add_node(z)

            self.log("add node for first ob ", ind)
            self.first_ob_index = ind
        assert ind != -1
        self.conn.send((1, ind))
        self.log("send finish")

    def recv_msg(self):
        # 0 —— retrieve q values
        # 1 —— peek or add node
        # 2 —— observe
        # 3 —— kill
        while self.conn.poll():
            # self.update_enough = False
            # self.iters_per_step = 0

            msg, obj = self.conn.recv()
            # self.log("receiving message", msg)
            if msg == 0:
                self.retrieve_q_value(obj)
            elif msg == 1:
                self.peek_node(obj)
            elif msg == 2:
                self.observe(obj)
            elif msg == 3:
                self.run_sweep = False
                self.conn.send((3, True))
            elif msg == 4:

                sampled = self.ec_buffer.sample(*obj)

                self.conn.send((4, sampled))
            elif msg == 5:
                indexes, z_new = obj
                self.ec_buffer.update(indexes, z_new)
                self.conn.send((5, True))
            elif msg == 6:
                indexes = obj

                self.conn.send((6, self.ec_buffer.states[indexes]))

            elif msg == 7:
                indexes = obj
                self.conn.send((7, np.nanmax(self.ec_buffer.external_value[indexes, :])))
            elif msg == 8:
                self.ec_buffer.recompute_density()
                self.conn.send((8, 0))
            elif msg == 9:
                buffer = LRU_KNN_GPU_PS_DENSITY if self.use_density else LRU_KNN_GPU_PS
                self.ec_buffer = buffer(self.buffer_size, self.latent_dim, 'game', 0, self.num_actions,
                                        self.knn)
                self.conn.send((9, 0))
            elif msg == 10:  # save
                filedir = obj
                self.save(filedir)
                self.conn.send((10, 0))
            elif msg == 11:  # load
                filedir = obj
                self.load(filedir)
                self.conn.send((11, 0))

            else:
                raise NotImplementedError
