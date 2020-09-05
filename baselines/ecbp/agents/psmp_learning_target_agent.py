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

from baselines.ecbp.agents.graph.build_graph_mer import *

from baselines import logger
import copy
import logging
from multiprocessing import Pipe
import cv2
import matplotlib.pyplot as plt

import pickle as pkl
import os

from baselines.ecbp.agents.graph.graph_util import *

class PSMPLearnTargetAgent(object):
    def __init__(self, model_func, exploration_schedule, obs_shape, vector_input=True, lr=1e-4, buffer_size=1000000,
                 num_actions=6, latent_dim=32,

                 gamma=0.99, knn=4, eval_epsilon=0.1, queue_threshold=5e-5, batch_size=32, density=True, trainable=True,
                 num_neg=10, debug=False,debug_dir = None,tf_writer=None):
        self.obs_shape = obs_shape
        self.debug=debug
        self.debug_dir = debug_dir
        self.conn, child_conn = Pipe()
        self.replay_buffer = np.empty((buffer_size + 10,) + obs_shape, np.float32 if vector_input else np.uint8)
        self.ec_buffer = PSLearningProcess(num_actions, buffer_size, latent_dim, obs_shape, child_conn, gamma,
                                           density=density)

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

        self.burnout = 10000000000
        self.update_target_freq = 10000
        self.buffer_capacity = 0
        self.trainable = trainable
        self.num_neg = num_neg
        self.loss_type = ["contrast"]
        input_type = U.Float32Input if vector_input else U.Uint8Input
        # input_type = U.Uint8Input
        self.hash_func, self.train_func, self.eval_func, self.norm_func, self.update_target_func = build_train_mer(
            input_type=input_type,
            obs_shape=obs_shape,

            model_func=model_func,
            num_actions=num_actions,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-4),
            gamma=gamma,
            grad_norm_clipping=10,
            latent_dim=latent_dim,

            loss_type=self.loss_type,
            batch_size=batch_size,
            num_neg=num_neg,
            c_loss_type="infonce",

        )
        self.finds = [0, 0]
        self.contrast_type = "both"
        self.augment_input_func, self.rand_init_func = build_random_input(input_type=input_type,
                                                                          obs_shape=obs_shape)
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


    def save(self, filedir, sess, saver):
        if not os.path.isdir(filedir):
            os.makedirs(filedir, exist_ok=True)
        self.send_and_receive(10, filedir)
        replay_buffer_file = open(os.path.join(filedir, "replay_buffer.pkl"), "wb")
        if np.size(self.replay_buffer)< 1024*1024*1024:
            pkl.dump((self.steps, self.replay_buffer, self.cur_capacity), replay_buffer_file)
        else:
            pkl.dump((self.steps, self.buffer_capacity), replay_buffer_file)
        model_file = os.path.join(filedir, "model_{}.pkl".format(self.steps))
        saver.save(sess, model_file)

    def load(self, filedir, sess, saver, num_steps):
        self.send_and_receive(11, filedir)
        if os.path.exists(os.path.join(filedir, "replay_buffer.pkl")):
            replay_buffer_file = open(os.path.join(filedir, "replay_buffer.pkl"), "rb")
            try:
                self.steps, self.replay_buffer, self.cur_capacity = pkl.load(replay_buffer_file)
            except ValueError:
                replay_buffer_file = open(os.path.join(filedir, "replay_buffer.pkl"), "rb")
                self.steps, self.buffer_capacity = pkl.load(replay_buffer_file)
        model_file = os.path.join(filedir, "model_{}.pkl".format(num_steps))
        for var_name, _ in tf.contrib.framework.list_variables(
                filedir):
            print(var_name)
        saver.restore(sess, model_file)

    def train(self):
        # sample
        # self.log("begin training")

        samples = self.send_and_receive(4, (self.batch_size, self.num_neg))
        samples_u = self.send_and_receive(4, (self.batch_size, self.num_neg))
        samples_v = self.send_and_receive(4, (self.batch_size, self.num_neg))
        index_u, _, _, _, value_u, _, _, _ = samples_u
        index_v, _, _, _, value_v, _, _, _ = samples_v
        index_tar, index_pos, index_neg, reward_tar, value_tar, action_tar, neighbours_index, neighbours_value = samples
        if len(index_tar) < self.batch_size:
            return

        obs_tar = [self.replay_buffer[ind] for ind in index_tar]
        # obs_pos = [self.replay_buffer[ind] for ind in index_pos]
        obs_neg = [self.replay_buffer[ind] for ind in index_neg]
        obs_neighbour = [self.replay_buffer[ind] for ind in neighbours_index]

        obs_u = [self.replay_buffer[ind] for ind in index_u]
        obs_v = [self.replay_buffer[ind] for ind in index_v]
        # print(obs_tar[0].shape)
        if self.contrast_type == "predictive":
            obs_pos = [self.replay_buffer[ind] for ind in index_pos]
        elif self.contrast_type == "augment":
            self.rand_init_func()
            obs_pos = self.augment_input_func(self.replay_buffer[index_tar])[0]
        elif self.contrast_type == "both":  # mixture
            self.rand_init_func()
            augment_inds = np.random.choice(self.batch_size, self.batch_size // 2)
            obs_pos = np.array([self.replay_buffer[ind] for ind in index_pos])
            obs_pos_augment = self.augment_input_func(self.replay_buffer[index_tar])[0]
            obs_pos[augment_inds] = obs_pos_augment[augment_inds]
        else:
            obs_pos =None
            raise NotImplementedError
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

        if "causality" in self.loss_type:
            input += [reward_tar, action_tar]
        if "weight_product" in self.loss_type:
            value_u = np.nan_to_num(np.array(value_u))
            value_v = np.nan_to_num(np.array(value_v))
            input += [obs_u, obs_v, obs_u, obs_v, value_u, value_v]

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

        self.send_and_receive(8, 0)  # recompute density
        self.log("finish updating target")

    def act(self, obs, is_train=True,debug=False):


        if is_train:
            self.steps += 1
            if self.steps % 100 == 0:
                self.log("steps", self.steps)

        # else:
        # self.log("obs", obs)

        # print(obs)
        self.obs = obs
        # print("in act",obs)
        self.z = self.hash_func(np.array(obs))
        self.z = np.array(self.z).reshape((self.latent_dim,))
        if is_train:

            if self.ind < 0 or self.ind >= self.buffer_capacity:
                self.ind = self.send_and_receive(1, (np.array(self.z), None))
                self.cur_capacity = max(self.ind, self.cur_capacity)
            # print(self.ind)
            self.replay_buffer[self.ind] = obs
            self.buffer_capacity = max(self.ind, self.buffer_capacity)

        # self.steps += 1
        epsilon = max(0, self.exploration_schedule.value(self.steps)) if is_train else self.eval_epsilon
        if np.random.random() < epsilon:
            self.log("Random action")
            action = np.random.randint(0, self.num_actions)
            return action
        else:
            # finds = np.zeros((1,))

            extrinsic_qs, intrinsic_qs, find,inds = self.send_and_receive(0, (np.array([self.z]), None, self.knn))
            extrinsic_qs, intrinsic_qs = np.array(extrinsic_qs), np.array(intrinsic_qs)
            inds = np.array(inds).reshape(-1)
            # print("debug? ",len(inds),debug)
            if len(inds) > 1 and debug:
                print("saving neightbour")
                self.save_neighbour(inds)
            self.finds[0] += sum(find)
            self.finds[1] += 1

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
            # print("action selection", max_action)
            # print("q", q, q_max)
            self.log("action selection", max_action)
            self.log("q", q, q_max)
            action_selected = np.random.randint(0, len(max_action))
            return int(max_action[action_selected])


    def save_neighbour(self,inds):
        save_path = os.path.join(self.debug_dir,"./neighbour/{}".format(self.steps))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for ind in inds:
            assert 0<= ind < self.cur_capacity
            cv2.imwrite(os.path.join(save_path,"{}.png".format(ind)),self.replay_buffer[ind].transpose(1,0,2))

    def empty_buffer(self):
        self.cur_capacity = 0
        self.steps = 0
        self.send_and_receive(9, 0)


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

            self.buffer_capacity = max(self.ind, self.buffer_capacity)

        else:
            self.ind = -1
            # self.ind = self.send_and_receive(1, (np.array([z_tp1]), None))

        if done:
            self.ind = -1

            if self.writer is not None:
                find_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="find rate", simple_value=self.finds[0] / (self.finds[1] + 1e-9))])
                self.writer.add_summary(find_summary, global_step=self.steps)

            self.finds = [0, 0]
            # self.steps = 0
        if self.steps > self.burnout:
            return

        if self.steps % self.train_step == 0 and self.steps >= self.burnin and train and self.trainable:
            self.train()
        if self.steps % self.update_target_freq == 0 and self.steps >= self.burnin and train and self.trainable:

            self.update_target()
        # else:
        #     self.log("not trai ning ", self.steps,self.steps % self.train_step == 0, self.steps >= self.burnin, train)

    # def update_sequence(self):
    #     self.ec_buffer.update_sequence(self.sequence, self.debug)
    #     self.sequence = []

    def finish(self):
        self.send_and_receive(3, (True,))
