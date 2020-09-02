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
from baselines.ecbp.agents.graph.build_graph_mer_attention import *
from baselines.ecbp.agents.graph.build_graph_dbc import *
from baselines import logger
import copy
import logging
from multiprocessing import Pipe
import cv2
import matplotlib.pyplot as plt
import pickle as pkl
import os
from baselines.ecbp.agents.psmp_learning_target_agent import PSMPLearnTargetAgent
import cv2


class MERAttentionAgent(PSMPLearnTargetAgent):
    def __init__(self, repr_func, model_func, exploration_schedule, obs_shape, vector_input=True, lr=1e-4,
                 buffer_size=1000000,
                 num_actions=6, latent_dim=32,
                 gamma=0.99, knn=4, eval_epsilon=0.1, queue_threshold=5e-5, batch_size=32, density=True, trainable=True,
                 num_neg=10, tf_writer=None):
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
        self.update_target_freq = 1000
        self.buffer_capacity = 0
        self.trainable = trainable
        self.num_neg = num_neg
        self.loss_type = ["contrast"]
        input_type = U.Float32Input if vector_input else U.Uint8Input
        # input_type = U.Uint8Input
        self.hash_func, self.train_func, self.eval_func, self.update_target_func = build_train_dbc(
            input_type=input_type,
            obs_shape=obs_shape,
            repr_func=repr_func,
            model_func=model_func,
            num_actions=num_actions,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-4),
            gamma=gamma,
            grad_norm_clipping=10,
            latent_dim=latent_dim,
            loss_type=self.loss_type,
            batch_size=batch_size,
            num_neg=num_neg,
            c_loss_type="margin",
        )
        self.finds = [0, 0]

        self.ec_buffer.start()

    def train(self):
        # sample
        # self.log("begin training")

        samples_u = self.send_and_receive(4, (self.batch_size, self.num_neg))
        samples_v = self.send_and_receive(4, (self.batch_size, self.num_neg))
        index_u, index_u_tp1, _, reward_u, _, action_u, _, _ = samples_u
        index_v, _, _, _, _, _, _, _ = samples_v
        if len(index_u) < self.batch_size:
            return
        obs_u = [self.replay_buffer[ind] for ind in index_u]
        obs_u_tp1 = [self.replay_buffer[ind] for ind in index_u_tp1]
        obs_v = [self.replay_buffer[ind] for ind in index_v]
        input = [obs_u, obs_u_tp1, obs_v, action_u, reward_u]
        func = self.train_func if self.steps < self.burnout else self.eval_func
        loss, summary = func(*input)
        # self.log("finish training")
        self.writer.add_summary(summary, global_step=self.steps)
