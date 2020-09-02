import numpy as np
from baselines.ecbp.agents.buffer.lru_knn_combine_bp import LRU_KNN_COMBINE_BP
from baselines.ecbp.agents.buffer.lru_knn_combine_bp_2 import LRU_KNN_COMBINE_BP_2
from baselines.ecbp.agents.buffer.lru_knn_prioritizedsweeping import LRU_KNN_PRIORITIZEDSWEEPING
from baselines.ecbp.agents.buffer.lru_knn_kbps import LRU_KNN_KBPS
from baselines.ecbp.agents.graph.model import *
import tensorflow as tf
from baselines.ecbp.agents.graph.build_graph_dueling import *
from baselines import logger
import copy
import logging


class KBPSAgent(object):
    def __init__(self, model_func, exploration_schedule, obs_shape, vector_input, lr=1e-4, buffer_size=1000000,
                 num_actions=6, latent_dim=32,
                 gamma=0.99, knn=10, eval_epsilon=0.01,
                 tf_writer=None, bp=True, debug=True):
        self.ec_buffer = LRU_KNN_KBPS(num_actions, buffer_size, latent_dim, latent_dim, gamma, bp, debug)
        self.obs = None
        self.z = None
        self.ind = -1
        self.writer = tf_writer
        self.sequence = []
        self.gamma = gamma
        self.bp = bp
        self.num_actions = num_actions
        self.exploration_schedule = exploration_schedule
        self.latent_dim = latent_dim
        self.knn = knn
        self.steps = 0
        self.rmax = 100000
        self.debug = debug
        self.logger = logging.getLogger("ecbp")
        self.heuristic_exploration = True
        self.eval_epsilon = eval_epsilon
        input_type = U.Float32Input if vector_input else U.Uint8Input
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

    def log(self, *args, logtype='debug', sep=' '):
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    def act(self, obs, is_train=True):
        self.obs = obs
        z = np.array(self.hash_func(np.array(obs))).reshape((self.latent_dim,))
        self.z = z
        if self.ind == -1:
            self.ind, knn_dist, knn_ind = self.ec_buffer.peek(z)
            knn_dist = np.array(knn_dist).reshape(-1).tolist()
            knn_ind = np.array(knn_ind).reshape(-1).tolist()
            if self.ind == -1:
                self.ind, _ = self.ec_buffer.ec_buffer.add_node(z)
                knn_dist = [0] + knn_dist
                knn_ind = [self.ind] + knn_ind
                if self.debug:
                    print("add node for first ob ", self.ind)
            self.ec_buffer.dist = knn_dist
            self.ec_buffer.ind = knn_ind
        else:
            ind, knn_dist, knn_ind = self.ec_buffer.peek(z)
            knn_dist = np.array(knn_dist).reshape(-1).tolist()
            knn_ind = np.array(knn_ind).reshape(-1).tolist()
            assert self.ind == ind, "{} {}".format(self.ind, ind)
            self.ec_buffer.dist = knn_dist
            self.ec_buffer.ind = knn_ind

        self.steps += 1
        epsilon = max(0, self.exploration_schedule.value(self.steps)) if is_train else self.eval_epsilon
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.num_actions)
            return action
        else:
            finds = np.zeros((1,))
            extrinsic_qs, intrinsic_qs, find = self.ec_buffer.ec_buffer.act_value(np.array([z]), self.knn)
            extrinsic_qs, intrinsic_qs = np.array(extrinsic_qs).squeeze(), np.array(intrinsic_qs).squeeze()
            finds += sum(find)
            if is_train:
                # self.log("extrinsic q shaps",extrinsic_qs.shape,extrinsic_qs)
                # extrinsic_qs = np.array([x if x > -self.rmax else 0 for x in extrinsic_qs])
                q = intrinsic_qs + extrinsic_qs
            else:
                q = extrinsic_qs
            q = np.squeeze(q)
            q_max = np.max(q)

            max_action = np.where(q >= q_max - 1e-7)[0]
            self.log("action selection", max_action)
            self.log("q", q, q_max)
            action_selected = np.random.randint(0, len(max_action))
            return max_action[action_selected]

    def observe(self, action, reward, state_tp1, done, train=True):

        z_tp1 = np.array(self.hash_func(np.array(state_tp1)[np.newaxis, ...])).reshape((self.latent_dim,))
        if train:
            self.ind = self.ec_buffer.prioritized_sweeping((self.ind, action, reward, z_tp1, done))
        else:
            self.ind, self.ec_buffer.dist, self.ec_buffer.ind = self.ec_buffer.peek(z_tp1)
        if done:
            self.ind = -1
            self.steps = 0

    def finish(self):
        pass
