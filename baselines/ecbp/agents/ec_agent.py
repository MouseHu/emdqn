import numpy as np
from baselines.ecbp.agents.buffer.lru_knn_combine_bp import LRU_KNN_COMBINE_BP
from baselines.ecbp.agents.buffer.lru_knn_count_gpu_fixmem import LRU_KNN_COUNT_GPU_FIXMEM
from baselines.ecbp.agents.graph.model import *
import tensorflow as tf
from baselines.ecbp.agents.graph.build_graph_dueling import *
from baselines import logger
from baselines.ecbp.agents.graph.build_graph_contrast_target import *

import copy
import logging


class ECAgent(object):
    def __init__(self, model_func, exploration_schedule, obs_shape, lr=1e-4, buffer_size=1000000,
                 num_actions=6, latent_dim=32,
                 gamma=0.99, knn=4,
                 tf_writer=None):
        self.ec_buffer = [LRU_KNN_COUNT_GPU_FIXMEM(buffer_size, latent_dim, "game", a, num_actions, knn) for a in
                          range(num_actions)]
        self.obs = None
        self.z = None
        self.state_tp1 = None
        self.writer = tf_writer
        self.sequence = []
        self.gamma = gamma
        self.num_actions = num_actions
        self.exploration_schedule = exploration_schedule
        self.latent_dim = latent_dim
        self.knn = knn
        self.steps = 0
        self.logger = logging.getLogger("ec")
        self.heuristic_exploration = False
        self.loss_type = ["contrast"]
        input_type = U.Uint8Input
        self.hash_func, _ , _ , _ , _ = build_train_contrast_target(
            make_obs_ph=lambda name: input_type(obs_shape, name=name),
            model_func=model_func,
            num_actions=num_actions,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-4),
            gamma=gamma,
            grad_norm_clipping=10,
            latent_dim=latent_dim,
            loss_type=self.loss_type
        )
        # self.hash_func, _, _ = build_train_dueling(
        #     make_obs_ph=lambda name: U.Uint8Input(obs_shape, name=name),
        #     model_func=model_func,
        #     q_func=model,
        #     imitate=False,
        #     num_actions=num_actions,
        #     optimizer=tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-4),
        #     gamma=gamma,
        #     grad_norm_clipping=10,
        # )
        self.finds = [0, 0]
        self.eval_epsilon = 0.01

    def log(self, *args, logtype='debug', sep=' '):
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    def act(self, obs, is_train=True):
        self.obs = obs
        z = np.array(self.hash_func(np.array(obs))).reshape((self.latent_dim,))
        self.z = z
        self.steps += 1
        # instance_inr = np.max(self.exploration_coef(self.count[obs]))
        epsilon = max(0, self.exploration_schedule.value(self.steps)) if is_train else self.eval_epsilon
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.num_actions)
            # print("random")
            return action
        else:
            extrinsic_qs = np.zeros((self.num_actions, 1))
            intrinsic_qs = np.zeros((self.num_actions, 1))
            finds = np.zeros((self.num_actions,))
            # print(self.num_actions)
            for a in range(self.num_actions):
                extrinsic_qs[a], intrinsic_qs[a], find = self.ec_buffer[a].act_value(np.array([z]), self.knn)
                finds[a] = find
            if is_train and self.heuristic_exploration:
                q = extrinsic_qs + intrinsic_qs
            else:
                q = extrinsic_qs
            self.finds[0] += sum(finds)
            self.finds[1] += self.num_actions
            q_max = np.max(q)
            max_action = np.where(q >= q_max - 1e-7)[0]
            action_selected = np.random.randint(0, len(max_action))
            self.log("capacity", [self.ec_buffer[a].curr_capacity for a in range(self.num_actions)])
            self.log("ec_action_selection", finds, q, q_max, max_action)
            return max_action[action_selected]

    def observe(self, action, reward, state_tp1, done, train=True):
        if not train:
            return
        self.sequence.append((copy.deepcopy(self.z), action, reward, done))
        self.state_tp1 = state_tp1
        if done:
            find_summary = tf.Summary(
                value=[tf.Summary.Value(tag="find rate", simple_value=self.finds[0] / (self.finds[1] + 1e-9))])
            self.writer.add_summary(find_summary, global_step=self.steps)
            self.finds = [0, 0]
        if done and train:
            self.update_sequence()

    def update_sequence(self):
        exRtn = [0]
        inRtn = [0]
        inrtd = 0
        for z, a, r, done in reversed(self.sequence):
            exrtd = self.gamma * exRtn[-1] + r
            inrtd = self.gamma * inRtn[-1] + inrtd

            exRtn.append(exrtd)
            inRtn.append(inrtd)
            q, inrtd = self.ec_buffer[a].peek(z, exrtd, inrtd, True)
            self.log("update sequence", q, exrtd, a)
            if q is None:  # new action
                self.ec_buffer[a].add(z, exrtd, inrtd)
                inrtd = self.ec_buffer[a].beta
        self.sequence = []
        return exRtn, inRtn
