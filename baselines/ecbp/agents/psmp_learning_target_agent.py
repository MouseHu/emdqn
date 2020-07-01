import numpy as np
from baselines.ecbp.agents.buffer.lru_knn_combine_bp import LRU_KNN_COMBINE_BP
from baselines.ecbp.agents.buffer.lru_knn_combine_bp_2 import LRU_KNN_COMBINE_BP_2
from baselines.ecbp.agents.buffer.prior_sweep_process import PriorSweepProcess
from baselines.ecbp.agents.buffer.ps_learning_process import PSLearningProcess
from baselines.ecbp.agents.buffer.ec_learning_process import ECLearningProcess
from baselines.ecbp.agents.graph.model import *
import tensorflow as tf
from baselines.ecbp.agents.graph.build_graph_dueling import *
from baselines.ecbp.agents.graph.build_graph_contrast_hash import *
from baselines.ecbp.agents.graph.build_graph_contrast_target import *
from baselines import logger
import copy
import logging
from multiprocessing import Pipe
import cv2
import matplotlib.pyplot as plt


class PSMPLearnTargetAgent(object):
    def __init__(self, model_func, exploration_schedule, obs_shape, vector_input=True, lr=1e-4, buffer_size=1000000,
                 num_actions=6, latent_dim=32,
                 gamma=0.99, knn=4, eval_epsilon=0.01, queue_threshold=5e-5, batch_size=32,
                 tf_writer=None):
        self.conn, child_conn = Pipe()
        self.replay_buffer = np.empty((buffer_size + 10,) + obs_shape, np.float32 if vector_input else np.uint8)
        self.ec_buffer = PSLearningProcess(num_actions, buffer_size, latent_dim, obs_shape, child_conn, gamma)
        self.obs = None
        self.z = None
        self.cur_capacity = 0
        self.ind = -1
        self.writer = tf_writer
        self.sequence = []
        self.gamma = gamma
        self.queue_threshold = queue_threshold
        self.num_actions = num_actions
        self.exploration_schedule = exploration_schedule
        self.latent_dim = latent_dim
        self.knn = knn
        self.steps = 0
        self.batch_size = batch_size
        self.rmax = 100000
        self.logger = logging.getLogger("ecbp")
        self.log("psmp learning agent here")
        self.eval_epsilon = eval_epsilon
        self.train_step = 4
        self.alpha = 1
        self.burnin = 2000
        self.burnout = 1000000
        self.update_target_freq = 10000
        self.loss_type = ["fit","contrast"]
        input_type = U.Float32Input if vector_input else U.Uint8Input
        # input_type = U.Uint8Input
        self.hash_func, self.train_func, self.eval_func, self.norm_func, self.update_target_func = build_train_contrast_target(
            make_obs_ph=lambda name: input_type(obs_shape, name=name),
            model_func=model_func,
            num_actions=num_actions,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-4),
            gamma=gamma,
            grad_norm_clipping=10,
            latent_dim=latent_dim,
            loss_type=self.loss_type
        )
        self.finds = [0, 0]

        self.ec_buffer.start()

    def log(self, *args, logtype='debug', sep=' '):
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    def send_and_receive(self, msg, obj):
        self.conn.send((msg, obj))
        # self.log("waiting")
        if self.conn.poll(timeout=None):
            recv_msg, recv_obj = self.conn.recv()
            assert msg == recv_msg
            return recv_obj

    def train(self):
        # sample
        # self.log("begin training")
        samples = self.send_and_receive(4, self.batch_size)
        index_tar, index_pos, index_neg, value_tar, action_tar, neighbours_index, neighbours_value = samples
        obs_tar = [self.replay_buffer[ind] for ind in index_tar]
        obs_pos = [self.replay_buffer[ind] for ind in index_pos]
        obs_neg = [self.replay_buffer[ind] for ind in index_neg]
        obs_neighbour = [self.replay_buffer[ind] for ind in neighbours_index]
        if "regression" in self.loss_type:
            value_original = self.norm_func(np.array(obs_tar))
            value_tar = np.array(value_tar)
            self.log(value_original, "value original")
            self.log(value_tar, "value tar")
            value_original = np.array(value_original).squeeze() / self.alpha
            assert value_original.shape == np.array(value_tar).shape, "{}{}".format(value_original.shape,
                                                                                    np.array(value_tar).shape)
            value_tar[np.isnan(value_tar)] = value_original[np.isnan(value_tar)]
            assert not np.isnan(value_tar).any(), "{}{}".format(value_original, obs_tar)
        input = [obs_tar]
        if "contrast" in self.loss_type:
            input += [obs_pos, obs_neg]
        if "regression" in self.loss_type:
            input += [np.nan_to_num(value_tar)]
        if "linear_model" in self.loss_type:
            input += [action_tar]
            if "contrast" not in self.loss_type:
                input += [obs_pos]
        if "fit" in self.loss_type:
            input += [obs_neighbour, np.nan_to_num(neighbours_value)]
            if "regression" not in self.loss_type:
                input += [np.nan_to_num(value_tar)]
        func = self.train_func if self.steps < self.burnout else self.eval_func
        loss, summary = func(*input)
        # self.log("finish training")
        self.writer.add_summary(summary, global_step=self.steps)

    def update_target(self):
        self.log("begin updating target")
        self.log("self.cur capacity", self.cur_capacity)
        self.update_target_func()
        for i in range(int(np.ceil((self.cur_capacity + 1) / self.batch_size))):
            low = i * self.batch_size
            high = min(self.cur_capacity + 1, (i + 1) * self.batch_size)
            self.log("low,high", low, high)
            obs_to_update = self.replay_buffer[low:high]
            # self.log("obs shape", obs_to_update.shape)
            z_to_update = self.hash_func(np.array(obs_to_update).astype(np.float32))
            # self.log("z shape", np.array(z_to_update).shape)
            self.send_and_receive(5, (np.arange(low, high), np.array(z_to_update)[0]))
        self.log("finish updating target")

    def act(self, obs, is_train=True):

        if is_train:
            self.steps += 1
            if self.steps % 100 == 0:
                self.log("steps", self.steps)
        else:
            self.log("obs",obs)
        # print(obs)
        self.obs = obs
        # print("in act",obs)
        self.z = self.hash_func(np.array(obs))
        self.z = np.array(self.z).reshape((self.latent_dim,))
        if is_train:
            if self.ind == -1:
                self.ind = self.send_and_receive(1, (np.array(self.z), None))
                self.cur_capacity = max(self.ind, self.cur_capacity)
            self.replay_buffer[self.ind] = obs
        # self.steps += 1
        epsilon = max(0, self.exploration_schedule.value(self.steps)) if is_train else self.eval_epsilon
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.num_actions)
            return action
        else:
            # finds = np.zeros((1,))
            extrinsic_qs, intrinsic_qs, find = self.send_and_receive(0, (np.array([self.z]), None, self.knn))
            extrinsic_qs, intrinsic_qs = np.array(extrinsic_qs), np.array(intrinsic_qs)
            self.finds[0] += sum(find)
            self.finds[1] += self.num_actions
            if is_train:
                q = extrinsic_qs
            else:
                q = extrinsic_qs

            q = np.squeeze(q)
            # q = np.nan_to_num(q)
            q_max = np.nanmax(q)
            if np.isnan(q_max):
                max_action = np.arange(self.num_actions)
            else:
                max_action = np.where(q >= q_max - 1e-7)[0]
            self.log("action selection", max_action)
            self.log("q", q, q_max)
            action_selected = np.random.randint(0, len(max_action))
            return int(max_action[action_selected])

    def observe(self, action, reward, state_tp1, done, train=True):
        if self.steps <= 1:
            self.update_target_func()
        z_tp1 = self.hash_func(np.array(state_tp1)[np.newaxis, ...])
        z_tp1 = np.array(z_tp1).reshape((self.latent_dim,))
        # z_tp1, h_tp1 = np.array(self.hash_func(np.array(state_tp1)[np.newaxis, ...])).reshape((self.latent_dim,))
        if train:
            self.ind = self.send_and_receive(2, (self.ind, action, reward, z_tp1, None, done))
            self.cur_capacity = max(self.ind, self.cur_capacity)
            self.replay_buffer[self.ind] = state_tp1
        else:
            self.ind = -1
            # self.ind = self.send_and_receive(1, (np.array([z_tp1]), None))

        if done:
            self.ind = -1

            find_summary = tf.Summary(
                value=[tf.Summary.Value(tag="find rate", simple_value=self.finds[0] / (self.finds[1] + 1e-9))])
            self.writer.add_summary(find_summary, global_step=self.steps)
            self.finds = [0, 0]
            # self.steps = 0
        if self.steps > self.burnout:
            return
        if self.steps % self.train_step == 0 and self.steps >= self.burnin and train:
            self.train()
        if self.steps % self.update_target_freq == 0 and self.steps >= self.burnin and train:
            self.update_target()
        # else:
        #     self.log("not trai ning ", self.steps,self.steps % self.train_step == 0, self.steps >= self.burnin, train)

    # def update_sequence(self):
    #     self.ec_buffer.update_sequence(self.sequence, self.debug)
    #     self.sequence = []

    def finish(self):
        self.send_and_receive(3, (True,))
