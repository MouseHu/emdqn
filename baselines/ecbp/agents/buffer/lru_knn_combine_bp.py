import numpy as np
from sklearn.neighbors import BallTree, KDTree
import os
from baselines.deepq.experiments.atari.lru_knn_gpu_bp import LRU_KNN_GPU_BP
import gc
from baselines.deepq.experiments.atari.knn_cuda_fixmem import knn as knn_cuda_fixmem
import copy


class LRU_KNN_COMBINE_BP(object):
    def __init__(self, num_actions, buffer_size, latent_dim, hash_dim, gamma=0.99, bp=True):
        self.ec_buffer = []
        self.num_actions = num_actions
        self.gamma = gamma
        self.rmax = 100000
        self.bp = bp
        for i in range(num_actions):
            self.ec_buffer.append(LRU_KNN_GPU_BP(buffer_size, latent_dim, hash_dim, 'game'))

    def add(self, action, key, value, reward, done, brothers):
        buffer = self.ec_buffer[action]
        if buffer.curr_capacity >= buffer.capacity:
            # find the LRU entry
            index = np.argmin(buffer.lru)
            self.ec_buffer[action].prev_id[index] = []
            self.ec_buffer[action].internal_value[index] = 0
            for bro in self.ec_buffer[action].brothers[index]:
                self.ec_buffer[bro[1]].brothers[bro[0]].remove((index, action))
            self.ec_buffer[action].brothers[index] = []
        else:
            index = buffer.curr_capacity
            buffer.curr_capacity += 1

        buffer.states[index] = key
        for a in range(len(brothers)):
            if brothers[a] > 0:
                buffer.brothers[index].append((brothers[a], a))
                self.ec_buffer[a].brothers[brothers[a]].append((index, action))
        buffer.external_value[index] = value
        buffer.reward[index] = reward
        buffer.done[index] = done
        buffer.lru[index] = buffer.tm
        buffer.tm += 0.01
        # print("here")
        # print(action, index, buffer.capacity)
        # print(buffer.address)
        key = np.array(key, copy=True)
        # key = copy.deepcopy(key)
        # print(key.shape)
        knn_cuda_fixmem.add(buffer.address, index, key)

        return index

    def act_value(self, keys, action, knn, bp):
        return self.ec_buffer[action].act_value(keys, knn, bp)

    def ancestor(self, state, action):
        sao_pair = copy.deepcopy(self.ec_buffer[action].prev_id[state])
        for bro in self.ec_buffer[action].brothers[state]:
            sao_pair_bro = self.ec_buffer[bro[1]].prev_id[bro[0]]
            for pair in sao_pair_bro:
                if pair not in sao_pair:
                    sao_pair.append(pair)
        # print("bro", (state, action))
        # print(self.ec_buffer[action].prev_id[state])
        # print(sa_pair)
        sa_pair = [(pair[0], pair[1]) for pair in sao_pair]
        return sa_pair

    def intrinsic_value(self, state, action):
        sa_pair = self.ec_buffer[action].brothers[state]
        sa_pair.append((state, action))
        actions = np.unique([a for s, a in sa_pair])
        # actions =
        if len(actions) < self.num_actions:
            return 0
        intrinsic_values = [self.ec_buffer[a].internal_value[s] for s, a in sa_pair]
        return np.max(intrinsic_values)

    def reward_update(self, stack):
        while len(stack) > 0:
            s0, s, a, r_i, r_e, d, r_loop = stack.pop(-1)
            old_r_e = copy.deepcopy(self.ec_buffer[a].external_value[s])
            # old_r_i = self.ec_buffer[a].internal_value[s]
            r = self.ec_buffer[a].reward[s]
            if self.bp:
                if s == s0 and d > 0:
                    # print("loop update", s, a, (r + self.gamma * r_loop) / (1 - self.gamma ** d),
                    #       self.ec_buffer[a].external_value[s])
                    self.ec_buffer[a].external_value[s] = max(self.ec_buffer[a].external_value[s],
                                                              (r + self.gamma * r_loop) / (1 - self.gamma ** d))
                    d = 0
                    r_loop = 0
                self.ec_buffer[a].external_value[s] = max(self.ec_buffer[a].external_value[s], r + self.gamma * r_e)
            # self.ec_buffer[a].internal_value[s] = min(self.ec_buffer[a].internal_value[s], r_i)
            if self.ec_buffer[a].external_value[s] > (old_r_e + 1e-7):
                # r_i = max([buffer.internal_value[s] for buffer in self.ec_buffer])
                # r_i = 0 if len(self.ancestor(s, a)) > 1 else r_i
                print("extrinsic update", s, a, self.ec_buffer[a].external_value[s], old_r_e)

                if d > 0:
                    r_loop = r_loop * self.gamma + r
                for sa_pair in self.ancestor(s, a):
                    stm1, atm1 = sa_pair
                    stack.append((s0, stm1, atm1, r_i, self.ec_buffer[a].external_value[s], d + 1, r_loop))

    def intrinsic_reward_update_iter(self, stack):
        while len(stack) > 0:
            s, a, v = stack.pop(-1)
            old_v_i = self.intrinsic_value(s, a)
            self.ec_buffer[a].internal_value[s] = min(v, self.ec_buffer[a].internal_value[s])
            v_i = self.intrinsic_value(s, a)
            if v_i < old_v_i and v_i < -self.rmax:
                for sa_pair in self.ancestor(s, a):
                    s_prev, a_prev = sa_pair
                    stack.append((s_prev, a_prev, v_i))

    def get_order(self, state, action, state_tp1, action_tp1):
        sao_pair = copy.deepcopy(self.ec_buffer[action_tp1].prev_id[state_tp1])
        for bro in self.ec_buffer[action_tp1].brothers[state_tp1]:
            sao_pair_bro = self.ec_buffer[bro[1]].prev_id[bro[0]]
            for pair in sao_pair_bro:
                if pair not in sao_pair:
                    sao_pair.append(pair)
        for s, a, order in sao_pair:
            if s == state and a == action:
                return order
        return -1

    def intrinsic_reward_update(self, sa_pair, sa_pair_tm1,debug=True):
        z_t, action_t, reward_t, z_tp1, done_t = sa_pair
        index = self.peek(z_t)
        if debug:
            print("intrinsic_reward_update_t", index)
        if index[action_t] < 0:
            ind_t = self.add(action_t, z_t, 0, reward_t, done_t, index)
            print("add",ind_t,action_t)
        else:
            ind_t = index[action_t]
        prev_s, prev_a = sa_pair_tm1
        if (prev_s, prev_a) not in self.ancestor(ind_t, action_t):
            order = len(self.ancestor(ind_t, action_t))
            self.ec_buffer[action_t].prev_id[ind_t].append((prev_s, prev_a, order))
        index_tp1 = self.peek(z_tp1)
        state_tp1, action_tp1 = np.max(index_tp1), np.argmax(index_tp1)
        if np.sqrt(np.sum(np.square(z_t - z_tp1))) < 1e-7:
            # self loop
            if debug:
                print("self loop")
            diminish_reward = -2 * self.rmax
        elif state_tp1 > 0:
            order = self.get_order(ind_t, action_t, state_tp1, action_tp1)
            ancestors = self.ancestor(state_tp1, action_tp1)
            if (state_tp1, action_tp1) in ancestors:
                # remove self loop
                ancestors.remove((state_tp1, action_tp1))
            if debug:
                print(ancestors)
                print("intrinsic update tp1 s{} ,a {},len(ancestors) {},order {}".format(state_tp1, action_tp1, len(ancestors), order), flush=True)
            diminish_reward = -2 * self.rmax if (len(ancestors) > 1 and order != 0) else 0
            diminish_reward = -2 * self.rmax if done_t else diminish_reward
        else:
            diminish_reward = -2 * self.rmax if done_t else 0
        if diminish_reward < 0:
            stack = [(ind_t, action_t, diminish_reward)]
            self.intrinsic_reward_update_iter(stack)

        return ind_t, action_t

    def peek(self, state):
        index = []
        for a in range(self.num_actions):
            ind = self.ec_buffer[a].peek(state)
            index.append(ind)
        return index

    def state_value(self, state):
        # TODO:optimize this using brothers
        act_values = []
        index = self.peek(state)
        for a in range(self.num_actions):
            ind = index[a]
            if ind > 0:
                act_values.append(self.ec_buffer[a].external_value[ind])
            else:
                act_values.append(-self.rmax)
        return np.max(act_values)

    def update_sequence(self, sequence, debug):
        Rtn = [0]
        state_index = []

        peek_count = 0
        for s, a, r, sp, done in reversed(sequence):
            if done or not self.bp:
                rtn = self.gamma * Rtn[-1] + r
            else:
                rtn = max(self.gamma * Rtn[-1] + r, self.gamma * self.state_value(sp) + r)
            Rtn.append(rtn)
            index = self.peek(s)
            ind = index[a]
            if ind < 0:
                if debug:
                    print("wierd")
                ind = self.add(a, s, rtn, r, done, index)
            else:
                peek_count += 1
                if self.ec_buffer[a].newly_added[ind]:
                    if debug:
                        print("sequence update new", ind, a, self.ec_buffer[a].external_value[ind], rtn)
                    self.ec_buffer[a].external_value[ind] = rtn
                    self.ec_buffer[a].newly_added[ind] = False
                else:
                    if debug:
                        print("sequence update", ind, a, max(self.ec_buffer[a].external_value[ind], rtn),
                              self.ec_buffer[a].external_value[ind])
                    self.ec_buffer[a].external_value[ind] = max(self.ec_buffer[a].external_value[ind], rtn)
            # self.ec_buffer[a].internal_value[ind] = min(self.ec_buffer[a].internal_value[ind], - done * self.rmax)
            state_index.append(ind)

        # prev_s, prev_a = None, None
        # for i, sample in enumerate(sequence):
        #     s, a, r, sp, done = sample
        #     ind = state_index[-i - 1]
        #     if prev_s is not None and (prev_s, prev_a) not in self.ec_buffer[a].prev_id[ind]:
        #         self.ec_buffer[a].prev_id[ind].append((prev_s, prev_a))
        #     prev_s, prev_a = ind, a

        print("peek count", peek_count / len(sequence))
        if self.bp:
            stack = []
            Rtn.pop()
            for i in range(len(sequence)):
                rtn = Rtn.pop()
                _, a, r, _, done = sequence[i]
                s = state_index[-i - 1]
                # print("put ",s,a)
                stack.append((s, s, a, (1 - done) * self.rmax, rtn, 0, 0))
                self.reward_update(stack)
