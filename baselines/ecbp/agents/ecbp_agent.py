import numpy as np
from baselines.ecbp.agents.buffer.lru_knn_combine_bp import LRU_KNN_COMBINE_BP
from baselines.ecbp.agents.graph.model import *
import tensorflow as tf
from baselines.ecbp.agents.graph.build_graph_dueling import *
from baselines import logger


class ECBPAgent(object):
    def __init__(self, model_func, exploration_schedule, obs_shape, lr=1e-4, buffer_size=1000000,
                 num_actions=6, latent_dim=32,
                 gamma=0.99, knn=4,
                 tf_writer=None, bp=True):
        self.ec_buffer = LRU_KNN_COMBINE_BP(num_actions, buffer_size, latent_dim, latent_dim, gamma, bp)
        self.obs = None
        self.z = None
        self.writer = tf_writer
        self.sequence = []
        self.gamma = gamma
        self.num_actions = num_actions
        self.exploration_schedule = exploration_schedule
        self.latent_dim = latent_dim
        self.knn = knn
        self.steps = 0
        self.hash_func, _, _ = build_train_dueling(
            make_obs_ph=lambda name: U.Uint8Input(obs_shape, name=name),
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
            print("random")
            return action
        else:
            extrinsic_qs = np.zeros((self.num_actions, 1))
            intrinsic_qs = np.zeros((self.num_actions, 1))
            finds = np.zeros((1,))
            # print(self.num_actions)
            for a in range(self.num_actions):
                print(a, end="-")
                extrinsic_qs[a], intrinsic_qs[a], find = self.ec_buffer.act_value(np.array([z]), a, self.knn)
                print(find,end="-")

                # print(" ")
                finds += sum(find)
            if is_train:
                q = extrinsic_qs + intrinsic_qs
            else:
                q = extrinsic_qs
            print(" ")
            print("train:", is_train, np.squeeze(extrinsic_qs))
            if is_train:
                print(np.squeeze(intrinsic_qs))
            q_max = np.max(q)
            max_action = np.where(q >= q_max - 1e-5)[0]
            action_selected = np.random.randint(0, len(max_action))
            return max_action[action_selected]

    def observe(self, action, reward, state_tp1, done, train=True):
        if not train:
            return
        z_tp1 = np.array(self.hash_func(np.array(state_tp1)[np.newaxis, ...])).reshape((self.latent_dim,))
        self.sequence.append((self.z, action, reward, z_tp1, done))

        if done:
            # if train:
            #     self.print_sequence()
            self.update_sequence()

    def update_sequence(self):
        self.ec_buffer.update_sequence(self.sequence)
        self.sequence = []
