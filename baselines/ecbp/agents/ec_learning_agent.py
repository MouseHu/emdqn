import numpy as np
# from baselines.ecbp.agents.buffer.lru_knn_combine_bp import LRU_KNN_COMBINE_BP
from baselines.ecbp.agents.buffer.lru_knn_combine import LRU_KNN_COMBINE
from baselines.ecbp.agents.graph.model import *
import tensorflow as tf
from baselines.ecbp.agents.graph.build_graph_dueling import *
from baselines import logger
from baselines.ecbp.agents.graph.build_graph_contrast_target import *
from baselines.ecbp.agents.graph.build_graph_mer_attention import *
from baselines.ecbp.agents.graph.graph_util import *

import copy
import logging


class ECLearningAgent(object):
    def __init__(self, model_func, exploration_schedule, obs_shape, vector_input=True, lr=1e-4, buffer_size=1000000,
                 num_actions=6, latent_dim=32,
                 gamma=0.99, knn=4, eval_epsilon=0.1, queue_threshold=5e-5, batch_size=32, density=True, trainable=True,
                 num_neg=10, debug=False, debug_dir=None,
                 tf_writer=None):
        self.gamma = gamma
        self.num_actions = num_actions
        self.exploration_schedule = exploration_schedule
        self.latent_dim = latent_dim
        self.knn = knn
        self.eval_epsilon = 0.02
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.writer = tf_writer
        self.trainable = trainable
        self.density = density
        self.debug_dir = debug_dir
        self.debug = debug
        self.eval_epsilon = eval_epsilon
        self.queue_threshold = queue_threshold
        self.obs_shape = obs_shape
        self.vector_input = vector_input
        self.buffer_size = buffer_size

        self.ec_buffer = LRU_KNN_COMBINE(num_actions, buffer_size, latent_dim, obs_shape, vector_input)
        self.obs = None
        self.z = None
        self.obs_tp1 = None
        self.steps = 0
        self.finds = [0, 0]
        self.burnin = 2000
        self.burnout = 100000000
        self.train_step = 4
        self.update_target_freq = 10000

        self.sequence = []

        self.cur_capacity = np.zeros(num_actions)
        self.logger = logging.getLogger("ec_agent")
        self.loss_type = ["contrast"]
        self.contrast_type = "predictive"

        input_type = U.Float32Input if vector_input else U.Uint8Input
        # input_type = U.Float32Input if vector_input else U.NormalizedUint8Input
        self.hash_func, self.train_func, self.eval_func, self.norm_func, self.attention_func, self.value_func, self.update_target_func = build_train_mer_attention(
            input_type=input_type,
            obs_shape=obs_shape,
            model_func=model_func,
            num_actions=num_actions,
            # optimizer=tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-4),
            optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr),
            gamma=gamma,
            grad_norm_clipping=10,
            latent_dim=latent_dim,
            loss_type=self.loss_type,
            batch_size=batch_size,
            num_neg=num_neg,
            c_loss_type="sqmargin",
        )

        self.augment_input_func, self.rand_init_func = build_random_input(input_type=input_type,
                                                                          obs_shape=obs_shape)

    def save(self, agentdir, sess, saver):
        pass

    def load(self, loaddir):
        pass

    def log(self, *args, logtype='debug', sep=' '):
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    def act(self, obs, is_train=True):

        self.obs = obs
        self.z = np.array(self.hash_func(np.array(obs))).reshape((self.latent_dim,))
        self.steps += 1
        epsilon = max(0, self.exploration_schedule.value(self.steps)) if is_train else self.eval_epsilon
        if np.random.random() < epsilon:  # epsilon greedy
            action = np.random.randint(0, self.num_actions)
            if not is_train:
                self.log("eval", "random", action)
            else:
                self.log("training", "random", action)
            return action
        else:
            extrinsic_qs = np.zeros((self.num_actions, 1))
            finds = np.zeros((self.num_actions,))
            for a in range(self.num_actions):
                extrinsic_qs[a], find = self.ec_buffer.act_value(a, np.array([self.z]), self.knn)
                finds[a] = find
            q = extrinsic_qs
            self.finds[0] += sum(finds)
            self.finds[1] += self.num_actions
            q_max = np.max(q)
            max_action = np.where(q >= q_max - 1e-7)[0]
            action_selected = np.random.randint(0, len(max_action))
            if not is_train:
                self.log("eval")
                self.log("capacity", self.ec_buffer.capacity())
                self.log("ec_action_selection", finds, q, q_max, max_action)
            return int(max_action[action_selected])

    def observe(self, action, reward, obs_tp1, done, train=True):
        if not train:
            return
        self.sequence.append((copy.deepcopy(self.obs), copy.deepcopy(self.z), action, reward, done))
        self.obs_tp1 = obs_tp1
        # print("obs diff", np.sum(np.abs(obs_tp1, self.obs)))
        if done:
            find_summary = tf.Summary(
                value=[tf.Summary.Value(tag="find rate", simple_value=self.finds[0] / (self.finds[1] + 1e-9))])
            self.writer.add_summary(find_summary, global_step=self.steps)
            self.finds = [0, 0]
        if done:
            self.ec_buffer.update_sequence(self.sequence, self.gamma)
            self.ec_buffer.update_kdtree()
            self.sequence = []

        if self.steps % self.train_step == 0 and self.steps >= self.burnin and train and self.trainable:
            self.train()
        if self.steps % self.update_target_freq == 0 and self.steps >= self.burnin and train and self.trainable:
            self.update_target()

    def train(self):
        # sample
        # self.log("begin training")

        samples = self.ec_buffer.sample(self.batch_size, self.num_neg)
        samples_u = self.ec_buffer.sample(self.batch_size, self.num_neg)
        samples_v = self.ec_buffer.sample(self.batch_size, self.num_neg)
        obs_u, _, _, value_u, _ = samples_u
        obs_v, _, _, value_v, _ = samples_v
        obs_tar, obs_pos, obs_neg, value_tar, action_tar = samples
        if len(obs_tar) < self.batch_size:
            return

        # print(obs_tar[0].shape)
        if self.contrast_type == "predictive":
            pass
        elif self.contrast_type == "augment":
            self.rand_init_func()
            obs_pos = self.augment_input_func(obs_pos)[0]
            self.rand_init_func()
            obs_tar = self.augment_input_func(obs_tar)[0]
            self.rand_init_func()
            obs_neg = self.augment_input_func(obs_neg)[0]
        elif self.contrast_type == "both":  # mixture
            augment_inds = np.random.choice(self.batch_size, self.batch_size // 2)

            self.rand_init_func()
            obs_pos[augment_inds] = self.augment_input_func(obs_pos)[0][augment_inds]

            self.rand_init_func()
            obs_tar[augment_inds] = self.augment_input_func(obs_tar)[0][augment_inds]

            self.rand_init_func()
            obs_neg[augment_inds] = self.augment_input_func(obs_neg)[0]
        else:
            raise NotImplementedError
        if "regression" in self.loss_type:
            value_original = self.norm_func(np.array(obs_tar))
            value_tar = np.array(value_tar)
            self.log(value_original, "value original")
            self.log(value_tar, "value tar")
            value_original = np.array(value_original).squeeze()
            assert value_original.shape == np.array(value_tar).shape, "{}{}".format(value_original.shape,
                                                                                    np.array(value_tar).shape)
            value_tar[np.isnan(value_tar)] = value_original[np.isnan(value_tar)]
            assert not np.isnan(value_tar).any(), "{}{}".format(value_original, obs_tar)
        input = [obs_tar]
        if "contrast" in self.loss_type:
            input += [obs_pos, obs_neg]
        if "regression" in self.loss_type:
            input += [np.nan_to_num(value_tar)]
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
        self.cur_capacity = self.ec_buffer.capacity()
        for a in range(self.num_actions):
            for i in range(int(np.ceil((self.cur_capacity[a] + 1) / self.batch_size))):
                low = i * self.batch_size
                high = min(self.cur_capacity[a] + 1, (i + 1) * self.batch_size)
                self.log("low,high", low, high)
                obs_to_update = self.ec_buffer.replay_buffer[low:high, a]
                # self.log("obs shape", obs_to_update.shape)
                z_to_update = self.hash_func(np.array(obs_to_update).astype(np.float32))
                # self.log("z shape", np.array(z_to_update).shape)
                self.ec_buffer.update(a*np.ones(high-low,dtype=np.int), np.arange(low, high), np.array(z_to_update)[0])
        self.ec_buffer.update_kdtree()
        self.log("finish updating target")

    def empty_buffer(self):
        del self.ec_buffer
        self.ec_buffer = LRU_KNN_COMBINE(self.num_actions, self.buffer_size, self.latent_dim, self.obs_shape,
                                         self.vector_input)
