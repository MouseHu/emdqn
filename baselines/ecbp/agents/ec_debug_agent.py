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
from baselines import logger
import copy
import logging
from multiprocessing import Pipe
from baselines.ecbp.agents.buffer.lru_knn_count_gpu_fixmem import LRU_KNN_COUNT_GPU_FIXMEM


class ECDebugAgent(object):
    def __init__(self, model_func, exploration_schedule, obs_shape, vector_input=True, lr=1e-4, buffer_size=1000000,
                 num_actions=6, latent_dim=32,
                 gamma=0.99, knn=4, eval_epsilon=0.01, queue_threshold=5e-5, batch_size=32,
                 tf_writer=None):
        self.conn, child_conn = Pipe()
        # self.replay_buffer = np.empty((buffer_size + 10,) + obs_shape, np.float32 if vector_input else np.uint8)
        self.ec_buffer = ECLearningProcess(num_actions, buffer_size, latent_dim, obs_shape, child_conn, gamma)
        self.obs = None
        self.z = None
        self.cur_capacity = 0
        self.ind = -1
        self.writer = tf_writer
        self.buffer_size = buffer_size
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
        self.sequence = []
        # self.ec_buffer = [
        #     LRU_KNN_COUNT_GPU_FIXMEM(self.buffer_size, self.latent_dim, "game", a, self.num_actions, self.knn) for a in
        #     range(self.num_actions)]
        # input_type = U.Float32Input if vector_input else U.Uint8Input
        # self.hash_func, self.train_func, self.eval_func, self.norm_func, self.update_target_func = build_train_contrast_target(
        #     make_obs_ph=lambda name: input_type(obs_shape, name=name),
        #     model_func=model_func,
        #     num_actions=num_actions,
        #     optimizer=tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-4),
        #     gamma=gamma,
        #     grad_norm_clipping=10,
        #     latent_dim=latent_dim,
        #     loss_type=self.loss_type
        # )
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
        self.ec_buffer.start()

    def log(self, *args, logtype='debug', sep=' '):
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    # def observe(self, sa_pair):
    #     z_t, index_t, action_t, reward_t, z_tp1, h_tp1, done_t = sa_pair
    #     self.sequence.append((copy.deepcopy(z_t), reward_t, action_t))
    #     if done_t:
    #         self.update_sequence()

    # def update_sequence(self):
    #     # to make sure that the final signal can be fast propagate through the state,
    #     # we need a sequence update like episodic control
    #     Rtn = 0
    #     for p, experience in enumerate(reversed(self.sequence)):
    #         z, r, a = experience
    #         Rtn = r + self.gamma * Rtn
    #
    #         q, _ = self.ec_buffer[a].peek(z, Rtn, 0, True)
    #         self.log("update sequence", q, Rtn, a)
    #         if q is None:  # new action
    #             self.ec_buffer[a].add(z, Rtn, self.rmax)
    #     self.sequence = []
    #     # self.ec_buffer.build_tree()

    # def send_and_receive(self, msg, obj):
    #     if msg == 0:
    #         return self.retrieve_q_value(obj)
    #     elif msg == 2:
    #         return self.observe_process(obj)
    #     else:
    #         raise NotImplementedError

    def send_and_receive(self, msg, obj):
        self.conn.send((msg, obj))
        self.log("waiting",msg)
        if self.conn.poll(timeout=None):
            self.log("recv here")
            recv_msg, recv_obj = self.conn.recv()
            assert msg == recv_msg
            return recv_obj

    # def retrieve_q_value(self, obj):
    #     z, knn = obj
    #     extrinsic_qs = np.zeros((self.num_actions, 1))
    #     intrinsic_qs = np.zeros((self.num_actions, 1))
    #     finds = np.zeros((self.num_actions,))
    #     for a in range(self.num_actions):
    #         extrinsic_qs[a], intrinsic_qs[a], find = self.ec_buffer[a].act_value(np.array([z]), knn)
    #         finds[a] = sum(find)
    #     self.log("capacity", [self.ec_buffer[a].curr_capacity for a in range(self.num_actions)])
    #     self.log("find", finds)
    #     return extrinsic_qs, intrinsic_qs, finds

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
        # if is_train:
        #     if self.ind == -1:
        #         self.ind = self.send_and_receive(1, (np.array([self.z]), None))
        #         self.cur_capacity = max(self.ind, self.cur_capacity)
        #     self.replay_buffer[self.ind] = obs
        # self.steps += 1
        epsilon = max(0, self.exploration_schedule.value(self.steps)) if is_train else self.eval_epsilon
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.num_actions)
            return action
        else:
            # finds = np.zeros((self.num_actions,))
            extrinsic_qs, intrinsic_qs, find = self.send_and_receive(0, (np.array([self.z]), None, self.knn))
            # extrinsic_qs, intrinsic_qs = np.array(extrinsic_qs), np.array(intrinsic_qs)
            # extrinsic_qs = np.zeros((self.num_actions, 1))
            # intrinsic_qs = np.zeros((self.num_actions, 1))
            # finds = np.zeros((self.num_actions,))
            # for a in range(self.num_actions):
            #     extrinsic_qs[a], intrinsic_qs[a], find = self.ec_buffer[a].act_value(np.array([self.z]), self.knn)
            #     finds[a] = sum(find)

            if is_train:
                q = extrinsic_qs
            else:
                q = extrinsic_qs

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

    # def act(self, obs, is_train=True):
    #     self.obs = obs
    #     z = np.array(self.hash_func(np.array(obs))).reshape((self.latent_dim,))
    #     self.z = z
    #     self.steps += 1
    #     # instance_inr = np.max(self.exploration_coef(self.count[obs]))
    #     epsilon = max(0, self.exploration_schedule.value(self.steps)) if is_train else self.eval_epsilon
    #     if np.random.random() < epsilon:
    #         action = np.random.randint(0, self.num_actions)
    #         # print("random")
    #         return action
    #     else:
    #         extrinsic_qs = np.zeros((self.num_actions, 1))
    #         intrinsic_qs = np.zeros((self.num_actions, 1))
    #         finds = np.zeros((self.num_actions,))
    #         # print(self.num_actions)
    #         for a in range(self.num_actions):
    #             extrinsic_qs[a], intrinsic_qs[a], find = self.ec_buffer[a].act_value(np.array([z]), self.knn)
    #             finds[a] = find
    #         # if is_train and self.heuristic_exploration:
    #         #     q = extrinsic_qs + intrinsic_qs
    #         # else:
    #         q = extrinsic_qs
    #
    #         q_max = np.max(q)
    #         max_action = np.where(q >= q_max - 1e-7)[0]
    #         action_selected = np.random.randint(0, len(max_action))
    #         self.log("capacity", [self.ec_buffer[a].curr_capacity for a in range(self.num_actions)])
    #         self.log("ec_action_selection", finds, q, q_max, max_action)
    #         return max_action[action_selected]

    # def observe(self, action, reward, state_tp1, done, train=True):
    #     if not train:
    #         return
    #     self.sequence.append((copy.deepcopy(self.z), action, reward, done))
    #     self.state_tp1 = state_tp1
    #     if done and train:
    #         self.update_sequence()

    # def update_sequence(self):
    #     exRtn = [0]
    #     inRtn = [0]
    #     inrtd = 0
    #     for z, a, r, done in reversed(self.sequence):
    #         exrtd = self.gamma * exRtn[-1] + r
    #         inrtd = self.gamma * inRtn[-1] + inrtd
    #
    #         exRtn.append(exrtd)
    #         inRtn.append(inrtd)
    #         q, inrtd = self.ec_buffer[a].peek(z, exrtd, inrtd, True)
    #         self.log("update sequence", q, exrtd, a)
    #         if q is None:  # new action
    #             self.ec_buffer[a].add(z, exrtd, inrtd)
    #             inrtd = self.ec_buffer[a].beta
    #     self.sequence = []
    #     return exRtn, inRtn

    def observe(self, action, reward, state_tp1, done, train=True):
        # z_tp1 = self.hash_func(np.array(state_tp1)[np.newaxis, ...])
        # z_tp1 = np.array(z_tp1).reshape((self.latent_dim,))
        # z_tp1, h_tp1 = np.array(self.hash_func(np.array(state_tp1)[np.newaxis, ...])).reshape((self.latent_dim,))
        if train:
            self.ind = self.send_and_receive(2, (self.z, self.ind, action, reward, None, None, done))
        else:
            self.ind = -1
            # self.ind = self.send_and_receive(1, (np.array([z_tp1]), None))

        if done:
            self.ind = -1
            # self.steps = 0
        if self.steps > self.burnout:
            return

    def finish(self):
        self.send_and_receive(3, (True,))
