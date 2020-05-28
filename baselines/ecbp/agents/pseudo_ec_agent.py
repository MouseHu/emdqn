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
from baselines.ecbp.agents.psmp_learning_target_agent import PSMPLearnTargetAgent
from baselines import logger
import copy
import logging
from multiprocessing import Pipe


class PseudoECAgent(PSMPLearnTargetAgent):
    def __init__(self, model_func, exploration_schedule, obs_shape, vector_input=True, lr=1e-4, buffer_size=1000000,
                 num_actions=6, latent_dim=32,
                 gamma=0.99, knn=4, eval_epsilon=0.01, queue_threshold=5e-5, batch_size=32,
                 tf_writer=None):
        self.conn, child_conn = Pipe()
        self.replay_buffer = np.empty((buffer_size + 10,) + obs_shape, np.float32 if vector_input else np.uint8)
        self.ec_buffer = ECLearningProcess(num_actions, buffer_size, latent_dim, obs_shape, child_conn, gamma)
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
        self.eval_epsilon = eval_epsilon
        self.train_step = 4
        self.alpha = 1
        self.burnin = 10000
        self.burnout = 10000000
        self.update_target_freq = 100000000
        self.loss_type = ["contrast"]
        input_type = U.Float32Input if vector_input else U.Uint8Input
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
        self.ec_buffer.start()

    def act(self, obs, is_train=True):
        if is_train:
            self.steps += 1
            if self.steps % 100 == 0:
                self.log("steps", self.steps)
        # print(obs)
        self.obs = obs
        # print("in act",obs)
        self.z = self.hash_func(np.array(obs))
        self.z = np.array(self.z).reshape((self.latent_dim,))
        if is_train:
            if self.ind == -1:
                self.ind = self.send_and_receive(1, (np.array([self.z]), None))
                self.cur_capacity = max(self.ind, self.cur_capacity)
            self.replay_buffer[self.ind] = obs
        # self.steps += 1
        epsilon = max(0, self.exploration_schedule.value(self.steps)) if is_train else self.eval_epsilon
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.num_actions)
            return action
        else:
            finds = np.zeros((1,))
            extrinsic_qs, intrinsic_qs, find = self.send_and_receive(0, (np.array([self.z]), None, self.knn))
            extrinsic_qs, intrinsic_qs = np.array(extrinsic_qs), np.array(intrinsic_qs)
            finds += sum(find)
            if is_train:
                q = extrinsic_qs
            else:
                q = extrinsic_qs

            q = np.squeeze(q)
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
            return max_action[action_selected]
