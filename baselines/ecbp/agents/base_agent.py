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


class BaseAgent(object):
    def __init__(self, model_func, ec_buffer, exploration_schedule, args, tf_writer=None):
        self.ec_buffer = ec_buffer
        self.obs = None
        self.z = None
        self.ind = -1
        self.writer = tf_writer
        self.sequence = []
        self.gamma = args.gamma
        self.num_actions = args.num_actions
        self.exploration_schedule = exploration_schedule
        self.latent_dim = args.latent_dim
        self.knn = args.knn
        self.steps = 0
        self.rmax = 100000
        self.logger = logging.getLogger(args.agent_name)
        self.eval_epsilon = args.eval_epsilon
        self.hash_func, _, _ = build_train_dueling(
            make_obs_ph=lambda name: U.Uint8Input(args.obs_shape, name=name),
            model_func=model_func,
            q_func=model,
            imitate=False,
            num_actions=args.num_actions,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-4),
            gamma=args.gamma,
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

            if self.ind == -1:
                self.ind, _ = self.ec_buffer.ec_buffer.add_node(z)
                knn_dist = [0] + knn_dist[1:]
                knn_ind = [self.ind] + knn_ind[1:]
                self.log("add node for first ob ", self.ind)
            self.ec_buffer.dist = knn_dist
            self.ec_buffer.ind = knn_ind
        self.steps += 1
        # instance_inr = np.max(self.exploration_coef(self.count[obs]))
        if (np.random.random() < max(0, self.exploration_schedule.value(self.steps))) and is_train:
            action = np.random.randint(0, self.num_actions)
            self.log("random")
            return action
        else:
            finds = np.zeros((1,))
            extrinsic_qs, intrinsic_qs, find = self.ec_buffer.ec_buffer.act_value(np.array([z]), self.knn)
            extrinsic_qs, intrinsic_qs = np.array(extrinsic_qs), np.array(intrinsic_qs)
            finds += sum(find)
            # if self.debug:
            #     print("old external q ", np.squeeze(extrinsic_qs), flush=True)
            if is_train:
                extrinsic_qs = np.array([x if x > -self.rmax else 0 for x in extrinsic_qs])
                q = intrinsic_qs + extrinsic_qs
            else:
                q = extrinsic_qs
            q = np.squeeze(q)
            q_max = np.max(q)

            max_action = np.where(q >= q_max - 1e-7)[0]
            self.log("action selection", max_action)
            self.log("q", q, q_max)
            action_selected = np.random.randint(0, len(max_action))
            # if self.debug:
            #     print("q ", q)
            #     print("q shape", q.shape, extrinsic_qs.shape, intrinsic_qs.shape)
            #     print("max action and random", max_action, action_selected)
            #     print("chosen action", max_action[action_selected])
            return max_action[action_selected]

    def observe(self, action, reward, state_tp1, done, train=True):
        pass
