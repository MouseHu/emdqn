import numpy as np
from baselines.ecbp.agents.buffer.lru_knn_combine_bp import LRU_KNN_COMBINE_BP
from baselines.ecbp.agents.buffer.lru_knn_combine_bp_2 import LRU_KNN_COMBINE_BP_2
from baselines.ecbp.agents.buffer.prior_sweep_process import PriorSweepProcess
from baselines.ecbp.agents.buffer.kbps_process import KernelBasedPriorSweepProcess
from baselines.ecbp.agents.graph.model import *
import tensorflow as tf
from baselines.ecbp.agents.graph.build_graph_dueling import *
from baselines.ecbp.agents.graph.build_graph_contrast_target import *

from baselines import logger
import copy
import logging
from multiprocessing import Pipe


class KBPSMPAgent(object):
    def __init__(self, model_func, exploration_schedule, obs_shape, vector_input, lr=1e-4, buffer_size=1000000,
                 num_actions=6, latent_dim=32,
                 gamma=0.99, knn=10, eval_epsilon=0.01, queue_threshold=1e-7, batch_size=32,
                 tf_writer=None):

        self.conn, child_conn = Pipe()
        self.queue_threshold = queue_threshold
        self.ec_buffer = KernelBasedPriorSweepProcess(num_actions, buffer_size, latent_dim, latent_dim, child_conn,
                                                      gamma)
        self.obs = None
        self.z = None
        self.ind = -1
        self.writer = tf_writer
        self.sequence = []
        self.gamma = gamma
        self.num_actions = num_actions
        self.exploration_schedule = exploration_schedule
        self.latent_dim = latent_dim
        self.knn = knn
        self.steps = 0
        self.rmax = 100000
        self.batch_size = batch_size
        self.logger = logging.getLogger("ecbp")
        self.eval_epsilon = eval_epsilon
        self.finds = [0, 0]
        input_type = U.Float32Input if vector_input else U.Uint8Input
        self.hash_func, self.train_func, self.eval_func, self.norm_func, self.update_target_func = build_train_contrast_target(
            make_obs_ph=lambda name: input_type(obs_shape, name=name),
            model_func=model_func,
            num_actions=num_actions,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-4),
            gamma=gamma,
            grad_norm_clipping=10
        )
        self.ec_buffer.start()

    def log(self, *args, logtype='debug', sep=' '):
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    def send_and_receive(self, msg, obj):
        self.conn.send((msg, obj))
        if self.conn.poll(timeout=None):
            recv_msg, recv_obj = self.conn.recv()
            assert msg == recv_msg
            return recv_obj

    def act(self, obs, is_train=True):
        self.obs = obs
        # print("in agent",obs)
        self.z = np.array(self.hash_func(np.array(obs))).reshape((self.latent_dim,))
        self.ind = self.send_and_receive(1, np.array([self.z]))
        self.steps += 1
        epsilon = max(0, self.exploration_schedule.value(self.steps)) if is_train else self.eval_epsilon
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.num_actions)
            return action
        else:
            # finds = np.zeros((1,))
            extrinsic_qs, intrinsic_qs, find = self.send_and_receive(0, (np.array([self.z]), self.knn))
            extrinsic_qs, intrinsic_qs = np.array(extrinsic_qs), np.array(intrinsic_qs)
            # finds += sum(find)
            if is_train:
                q = intrinsic_qs + extrinsic_qs
            else:
                q = extrinsic_qs
            self.finds[0] += sum(find)
            self.finds[1] += self.num_actions
            q = np.squeeze(q)
            q_max = np.nanmax(q)
            if np.isnan(q_max):
                max_action = np.arange(self.num_actions)
            else:
                max_action = np.where(q >= q_max - 1e-7)[0]
            self.log("action selection", max_action)
            self.log("q", q, q_max)
            action_selected = np.random.randint(0, len(max_action))
            return max_action[action_selected]

    def observe(self, action, reward, state_tp1, done, train=True):
        z_tp1 = np.array(self.hash_func(np.array(state_tp1)[np.newaxis, ...])).reshape((self.latent_dim,))
        if train:
            self.ind = self.send_and_receive(2, (self.ind, action, reward, z_tp1, done))
        else:
            self.ind= -1
            # self.ind = self.send_and_receive(1, np.array([z_tp1]))

            # self.ind = self.send_and_receive(1, (np.array([z_tp1]), None))
        if done:
            find_summary = tf.Summary(
                value=[tf.Summary.Value(tag="find rate", simple_value=self.finds[0] / (self.finds[1] + 1e-9))])
            self.writer.add_summary(find_summary, global_step=self.steps)
            self.finds = [0, 0]
            self.ind = -1
            # self.steps = 0

    # def update_sequence(self):
    #     self.ec_buffer.update_sequence(self.sequence, self.debug)
    #     self.sequence = []

    def finish(self):
        self.send_and_receive(3, (True,))
        self.ec_buffer.join()
