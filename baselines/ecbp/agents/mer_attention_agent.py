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
from baselines.ecbp.agents.graph.graph_util import *
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
    def __init__(self, model_func, exploration_schedule, obs_shape, vector_input=True, lr=1e-4, buffer_size=1000000,
                 num_actions=6, latent_dim=32,
                 gamma=0.99, knn=4, eval_epsilon=0.1, queue_threshold=5e-5, batch_size=32, density=True, trainable=True,
                 num_neg=10, debug=False,debug_dir = None,tf_writer=None):

        self.debug=debug
        self.debug_dir = debug_dir
        self.obs_shape = obs_shape
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
        self.loss_type = ["attention", "contrast"]
        input_type = U.Float32Input if vector_input else U.Uint8Input
        # input_type = U.Uint8Input
        self.hash_func, self.train_func, self.eval_func, self.norm_func, self.attention_func, self.value_func, self.update_target_func = build_train_mer_attention(
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
        self.contrast_type = "predictive"
        self.augment_input_func,self.rand_init_func = build_random_input(input_type=input_type,
                                                     obs_shape=obs_shape)
        self.finds = [0, 0]

        self.ec_buffer.start()

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
        obs_neg = [self.replay_buffer[ind] for ind in index_neg]
        obs_neighbour = [self.replay_buffer[ind] for ind in neighbours_index]
        obs_u = [self.replay_buffer[ind] for ind in index_u]
        obs_v = [self.replay_buffer[ind] for ind in index_v]

        if self.contrast_type == "both":
            obs_pos = [self.replay_buffer[ind] for ind in index_pos]
        elif self.contrast_type == "augment":
            self.rand_init_func()
            obs_pos = self.augment_input_func(self.replay_buffer[index_tar])[0]
        else:  # mixture
            self.rand_init_func()
            augment_inds = np.random.choice(self.batch_size, self.batch_size // 2)
            obs_pos = np.array([self.replay_buffer[ind] for ind in index_pos])
            obs_pos_augment = self.augment_input_func(self.replay_buffer[index_tar])[0]
            obs_pos[augment_inds] = obs_pos_augment[augment_inds]

        # print(obs_tar[0].shape)
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
        if "attention" in self.loss_type:
            value_original = self.value_func(np.array(obs_tar))
            value_tar = np.array(value_tar)
            value_original = np.array(value_original).squeeze()
            value_tar[np.isnan(value_tar)] = value_original[np.isnan(value_tar)]
            input += [value_tar]
        func = self.train_func if self.steps < self.burnout else self.eval_func
        loss, summary = func(*input)
        # self.log("finish training")
        self.writer.add_summary(summary, global_step=self.steps)

    def save_attention(self, filedir, step):
        subdir = os.path.join(filedir, "./attention")
        attention = np.array(self.attention_func(np.array(self.obs)))
        value = np.array(self.value_func(np.array(self.obs)))
        print("var", np.var(attention), np.max(attention), np.min(attention), value)
        # print(attention.squeeze())


        attention = attention.reshape(28, 28)
        attention = (attention - np.min(attention)) / (np.max(attention) - np.min(attention))
        attention = cv2.resize(attention, (220, 220))
        attention = np.repeat(attention[..., np.newaxis], 3, axis=2)
        # attention[1:, ...] = 1
        attentioned_image = np.array(self.obs)[0] * attention
        if not os.path.isdir(subdir):
            os.makedirs(os.path.join(subdir, "./mask/"))
            os.makedirs(os.path.join(subdir, "./image/"))
        # print(attention.shape)
        cv2.imwrite(os.path.join(subdir, "./mask/", "masked_image_{}.png".format(step)),
                    attentioned_image.transpose((1, 0, 2)))
                    # attentioned_image)
        cv2.imwrite(os.path.join(subdir, "./image/", "attention_{}.png".format(step)),
                    # attention * 255)
        attention.transpose((1, 0, 2)) * 255)
