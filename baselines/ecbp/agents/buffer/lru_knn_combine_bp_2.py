import numpy as np
from sklearn.neighbors import BallTree, KDTree
import os
from baselines.ecbp.agents.buffer.lru_knn_gpu_bp_2 import LRU_KNN_GPU_BP_2
import gc
from baselines.deepq.experiments.atari.knn_cuda_fixmem import knn as knn_cuda_fixmem
import copy


class LRU_KNN_COMBINE_BP_2(object):
    def __init__(self, num_actions, buffer_size, latent_dim, hash_dim, gamma=0.99, bp=True, debug=True):
        self.num_actions = num_actions
        self.gamma = gamma
        self.rmax = 100000
        self.bp = bp
        self.debug = debug
        self.ec_buffer = LRU_KNN_GPU_BP_2(buffer_size, latent_dim, hash_dim, 'game', num_actions, debug=debug)

    # def act_value(self, keys, action, knn):
    #     return self.ec_buffer.act_value(keys, knn)

    def reward_update_iter(self, stack):
        while len(stack) > 0:
            s0, s, a, r_i, r_e, d, r_loop = stack.pop(-1)
            old_r_e = np.max(self.ec_buffer.external_value[s, :])
            # old_r_i = self.ec_buffer[a].internal_value[s]
            r = self.ec_buffer.reward[s, a]
            if self.bp and old_r_e > -self.rmax:
                if s == s0 and d > 0:
                    # print("loop update", s, a, (r + self.gamma * r_loop) / (1 - self.gamma ** d),
                    #       self.ec_buffer[a].external_value[s])
                    self.ec_buffer.external_value[s, a] = max(self.ec_buffer.external_value[s, a],
                                                              (r + self.gamma * r_loop) / (1 - self.gamma ** d))
                    d = 0
                    r_loop = 0
                self.ec_buffer.external_value[s, a] = max(self.ec_buffer.external_value[s, a], r + self.gamma * r_e)
            # self.ec_buffer[a].internal_value[s] = min(self.ec_buffer[a].internal_value[s], r_i)

            new_r_e = np.max(self.ec_buffer.external_value[s, :])
            if new_r_e > (old_r_e + 1e-7):
                # r_i = max([buffer.internal_value[s] for buffer in self.ec_buffer])
                # r_i = 0 if len(self.ancestor(s, a)) > 1 else r_i
                print("extrinsic update", s, a, self.ec_buffer.external_value[s, a], old_r_e)

                if d > 0:
                    r_loop = r_loop * self.gamma + r
                for sa_pair in self.ec_buffer.prev_id[s]:
                    stm1, atm1 = sa_pair
                    stack.append((s0, stm1, atm1, r_i, new_r_e, d + 1, r_loop))

    def intrinsic_reward_update_iter(self, stack):
        while len(stack) > 0:
            s, a, v = stack.pop(-1)
            old_v_i = np.max(self.ec_buffer.internal_value[s, :])
            old_v_a = self.ec_buffer.internal_value[s, a]
            self.ec_buffer.internal_value[s, a] = min(v, self.ec_buffer.internal_value[s, a])
            v_i = np.max(self.ec_buffer.internal_value[s, :])
            print("lower internal reward", s, a, v, old_v_a)
            if v_i < old_v_i:

                for sa_pair in self.ec_buffer.prev_id[s]:
                    s_prev, a_prev = sa_pair
                    stack.append((s_prev, a_prev, v_i))

    def intrinsic_reward_update(self, sa_pair, debug=True, cut_done=True):
        index_t, action_t, reward_t, z_tp1, done_t = sa_pair
        index_tp1 = self.peek(z_tp1)

        if index_tp1 < 0:
            index_tp1 = self.ec_buffer.add_node(z_tp1)
            if debug:
                print("add node", index_tp1)

        if (index_t, action_t) not in self.ec_buffer.prev_id[index_tp1]:
            if debug:
                print("add edge", index_t, action_t, index_tp1)
            self.ec_buffer.add_edge(index_t, index_tp1, action_t, reward_t, done_t, -self.rmax, 0)

        if index_t == index_tp1:
            # self loop
            if debug:
                print("self loop")
            diminish_reward = -2 * self.rmax
        elif done_t and cut_done:
            diminish_reward = -2 * self.rmax
        else:
            diminish_reward = 0
            ancestors = copy.deepcopy(self.ec_buffer.prev_id[index_tp1])
            for pair in ancestors:
                s, a = pair
                if s == index_tp1:
                    # contain self loop
                    ancestors.remove(pair)
                    break
            for i, pair in enumerate(ancestors):
                s, a = pair
                if index_t == s and action_t == a and i >= 1:
                    # cut edge
                    if debug:
                        print("cut edge")
                        print(ancestors)
                    diminish_reward = -2 * self.rmax
                    break

        if diminish_reward < 0:
            stack = [(index_t, action_t, diminish_reward)]
            self.intrinsic_reward_update_iter(stack)

        return index_tp1

    def peek(self, state):
        ind = self.ec_buffer.peek(state)
        return ind

    def reward_update(self, stack):
        while len(stack) > 0:
            s0, s, a, r_e, d, r_loop = stack.pop(-1)
            old_r_e = self.ec_buffer.external_value[s, a]
            # old_r_i = self.ec_buffer[a].internal_value[s]
            r = self.ec_buffer.reward[s, a]
            if self.bp:
                if s == s0 and d > 0:
                    self.ec_buffer.external_value[s, a] = max(self.ec_buffer.external_value[s, a],
                                                              (r + self.gamma * r_loop) / (1 - self.gamma ** d))
                    d = 0
                    r_loop = 0
                self.ec_buffer.external_value[s, a] = max(self.ec_buffer.external_value[s, a], r + self.gamma * r_e)
            self.ec_buffer.newly_added[s, a] = False
            if self.ec_buffer.external_value[s, a] > (old_r_e + 1e-7):
                # print("extrinsic update", s, a, self.ec_buffer.external_value[s, a], old_r_e)

                if d > 0:
                    r_loop = r_loop * self.gamma + r
                for sa_pair in self.ec_buffer.prev_id[s]:
                    stm1, atm1 = sa_pair
                    if not self.ec_buffer.newly_added[stm1, atm1]:
                        stack.append((s0, stm1, atm1, np.max(self.ec_buffer.external_value[s, :]), d + 1, r_loop))

    def state_value(self, index):
        if index > 0:
            external_values = self.ec_buffer.external_value[index, :]
            internal_values = self.ec_buffer.internal_value[index, :]
        else:
            external_values = -self.rmax * np.ones(self.num_actions)
            internal_values = np.zeros(self.num_actions)
        return np.max(external_values), np.max(internal_values)

    def update_sequence(self, sequence, debug):
        Rtn = [0]
        peek_count = 0
        for s, a, r, sp, done in reversed(sequence):
            rtn = np.max(self.gamma * self.ec_buffer.external_value[sp, :] + r)
            # if debug:
            #     print("update sequence ", s, a, r, sp, done)
            #     print("old rtn", rtn)
            if done and r == 0:
                Rtn[0] = self.state_value(sp)[0]

            if s < 0:
                if debug:
                    print("wierd")
                s = self.ec_buffer.add_node(s)
            else:
                peek_count += 1
            if self.ec_buffer.newly_added[s, a]:
                stack = [(s, s, a, Rtn[-1], 0, 0)]
                self.reward_update(stack)
            rtn = np.max(self.ec_buffer.external_value[s, :])
            # if debug:
            #     print("new return", rtn)
            Rtn.append(rtn)
