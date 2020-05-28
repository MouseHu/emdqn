import numpy as np
from baselines.ecbp.agents.buffer.lru_knn_combine import LRU_KNN_COMBINE
from baselines.ecbp.agents.graph.model import *
import tensorflow as tf
from baselines.ecbp.agents.graph.build_graph_dueling import *
from baselines import logger
import copy


class OnlineAgent(object):
    def __init__(self, model_func, exploration_schedule, obs_shape, input_type, lr=1e-4, buffer_size=1000000,
                 num_actions=6, latent_dim=32,
                 gamma=0.99, knn=4,
                 tf_writer=None):
        self.ec_buffer = LRU_KNN_COMBINE(num_actions, buffer_size, latent_dim, latent_dim, obs_shape, 'game')
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
        self.heuristic_exploration = True
        self.hash_func, _, _ = build_train_dueling(
            make_obs_ph=lambda name: input_type(obs_shape, name=name),
            model_func=model_func,
            q_func=model,
            imitate=False,
            num_actions=num_actions,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-4),
            gamma=gamma,
            grad_norm_clipping=10,
        )

    def act(self, obs, is_train=True):
        self.obs = obs
        z = np.array(self.hash_func(np.array(obs))).reshape((self.latent_dim,))
        self.z = z
        self.steps += 1
        # instance_inr = np.max(self.exploration_coef(self.count[obs]))
        if (np.random.random() < max(0, self.exploration_schedule.value(self.steps))) and is_train:
            action = np.random.randint(0, self.num_actions)
            # print("random")
            return action
        else:
            extrinsic_qs = np.zeros((self.num_actions, 1))
            intrinsic_qs = np.zeros((self.num_actions, 1))
            finds = np.zeros((1,))
            # print(self.num_actions)
            for a in range(self.num_actions):
                extrinsic_qs[a], intrinsic_qs[a], find = self.ec_buffer[a].act_value(np.array([z]), self.knn)
                finds += sum(find)
            if is_train and self.heuristic_exploration:
                q = extrinsic_qs + intrinsic_qs
            else:
                q = extrinsic_qs

            q_max = np.max(q)
            max_action = np.where(q >= q_max - 1e-7)[0]
            action_selected = np.random.randint(0, len(max_action))
            return max_action[action_selected]

    def observe(self, action, reward, state_tp1, done, train=True):
        if not train:
            return
        self.sequence.append((copy.deepcopy(self.z), action, reward, done))
        self.state_tp1 = state_tp1
        if done and train:
            self.update_sequence()

    def train(self):
        pass

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
            if q is None:  # new action
                self.ec_buffer[a].add(z, exrtd, inrtd)
                inrtd = self.ec_buffer[a].beta
        self.sequence = []
        return exRtn, inRtn
