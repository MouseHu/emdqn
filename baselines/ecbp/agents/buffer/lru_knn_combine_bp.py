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

    def act_value(self, keys, action, knn):
        return self.ec_buffer[action].act_value(keys, knn)

    def ancestor(self, state, action):
        sa_pair = copy.deepcopy(self.ec_buffer[action].prev_id[state])
        for bro in self.ec_buffer[action].brothers[state]:
            sa_pair_bro = self.ec_buffer[bro[1]].prev_id[bro[0]]
            for pair in sa_pair_bro:
                if pair not in sa_pair:
                    sa_pair.append(pair)
        # print("bro", (state, action))
        # print(self.ec_buffer[action].prev_id[state])
        # print(sa_pair)
        return sa_pair

    def reward_update(self, stack):
        while len(stack) > 0:
            s0, s, a, r_i, r_e, d, r_loop = stack.pop(-1)
            old_r_e = self.ec_buffer[a].external_value[s]
            old_r_i = self.ec_buffer[a].internal_value[s]
            r = self.ec_buffer[a].reward[s]
            if self.bp:
                if s == s0 and d > 0:
                    # print("circle", self.ec_buffer[a].external_value[s],
                    #       (r + self.gamma * r_loop) / (1 - self.gamma ** d))
                    self.ec_buffer[a].external_value[s] = max(self.ec_buffer[a].external_value[s],
                                                              (r + self.gamma * r_loop) / (1 - self.gamma ** d))
                    d = 0
                    r_loop = 0
                self.ec_buffer[a].external_value[s] = max(self.ec_buffer[a].external_value[s], r + self.gamma * r_e)
            # print("r",r)
            # print("depth ", d)
            self.ec_buffer[a].internal_value[s] = min(self.ec_buffer[a].internal_value[s], r_i)
            if self.ec_buffer[a].internal_value[s] < old_r_i or self.ec_buffer[a].external_value[s] > (old_r_e + 1e-7):
                # print("updated ", self.ec_buffer[a].external_value[s], old_r_e)
                # print("r", r)
                # if self.ec_buffer[a].external_value[s] > 1:
                #     print(s,a,self.ec_buffer[a].external_value[s] )
                #     print("????")
                #     self.ancestor(s, a)
                #     exit(-1)
                r_i = max([buffer.internal_value[s] for buffer in self.ec_buffer])
                if d > 0:
                    r_loop = r_loop * self.gamma + r
                for sa_pair in self.ancestor(s, a):
                    # print(sa_pair, d, "debug")
                    stm1, atm1 = sa_pair
                    if self.ec_buffer[atm1].reward[stm1] == 1:
                        print("bug", s, a)
                    stack.append((s0, stm1, atm1, r_i, self.ec_buffer[a].external_value[s], d + 1, r_loop))
                    # self.reward_update(stack)

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

    def update_sequence(self, sequence):
        Rtn = [0]
        state_index = []

        peek_count = 0
        for s, a, r, sp, done in reversed(sequence):
            if done:
                rtn = self.gamma * Rtn[-1] + r
            else:
                rtn = max(self.gamma * Rtn[-1] + r, self.gamma * self.state_value(sp) + r)
            Rtn.append(rtn)
            # print(rtn)
            # self.table[s, a] = max(self.table[s, a], Rtn[-1])
            index = self.peek(s)
            ind = index[a]
            if ind < 0:
                ind = self.add(a, s, Rtn[-1], r, done, index)
            else:
                peek_count += 1
                # print(a,self.ec_buffer[a].external_value[ind], Rtn[-1])
                self.ec_buffer[a].external_value[ind] = max(self.ec_buffer[a].external_value[ind], Rtn[-1])
            self.ec_buffer[a].internal_value[ind] = min(self.ec_buffer[a].internal_value[ind], - done * self.rmax)
            state_index.append(ind)
            # if r == 1:
            #     print("goal", ind, a)
            # else:
            #     print("sequence", ind, a)
            # if prev_s is not None and (prev_s, prev_a) not in self.ec_buffer[a].prev_id[ind]:
            #     self.ec_buffer[a].prev_id[ind].append((prev_s, prev_a))
            # print("prev", ind, a)
            # print(self.ec_buffer[a].prev_id[ind])
            # prev_s, prev_a = ind, a
        prev_s, prev_a = None, None
        for i, sample in enumerate(sequence):
            s, a, r, sp, done = sample
            ind = state_index[-i - 1]
            if prev_s is not None and (prev_s, prev_a) not in self.ec_buffer[a].prev_id[ind]:
                self.ec_buffer[a].prev_id[ind].append((prev_s, prev_a))
            prev_s, prev_a = ind, a

        print("peek count", peek_count / len(sequence))
        # prev_s = None
        # for s, a, r, _, done in reversed(sequence):
        #     if prev_s is not None:
        #         print(np.linalg.norm(s - prev_s))
        #     prev_s = s
        stack = []
        Rtn.pop()
        for i in range(len(sequence)):
            rtn = Rtn.pop()
            _, a, r, _, done = sequence[i]
            s = state_index[-i-1]
            # print("put ",s,a)
            stack.append((s, s, a, (1 - done) * self.rmax, rtn, 0, 0))
            self.reward_update(stack)
