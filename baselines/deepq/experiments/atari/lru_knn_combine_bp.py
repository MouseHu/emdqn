import numpy as np
from sklearn.neighbors import BallTree, KDTree
import os
from baselines.deepq.experiments.atari.lru_knn_gpu_bp import LRU_KNN_GPU_BP
import gc
from baselines.deepq.experiments.atari.knn_cuda_fixmem import knn as knn_cuda_fixmem


class LRU_KNN_COMBINE_BP(object):
    def __init__(self, num_actions, buffer_size, latent_dim, hash_dim, gamma=0.99):
        self.ec_buffer = []
        self.num_actions = num_actions
        self.gamma = gamma
        self.rmax = 10000000000
        for i in range(num_actions):
            self.ec_buffer.append(LRU_KNN_GPU_BP(buffer_size, latent_dim, hash_dim, 'game'))

    def add(self, action, key, value, reward, done):
        buffer = self.ec_buffer[action]
        if buffer.curr_capacity >= buffer.capacity:
            # find the LRU entry
            index = np.argmin(buffer.lru)
            self.ec_buffer[action].prev_id[index] = []
            self.ec_buffer[action].internal_value[index] = 0
        else:
            index = buffer.curr_capacity
            buffer.curr_capacity += 1

        buffer.states[index] = key
        buffer.external_value[index] = value
        buffer.reward[index] = reward
        buffer.done[index] = done
        buffer.lru[index] = buffer.tm
        buffer.tm += 0.01
        return index

    def act_value(self, keys, action, knn):
        return self.ec_buffer[action].act_value(keys, knn)

    def reward_update(self, s0, s, a, r_i, r_e, d, r_loop):
        old_r_e = self.ec_buffer[a].external_value[s]
        old_r_i = self.ec_buffer[a].internal_value[s]
        r = self.ec_buffer[a].reward[s]

        if s == s0 and d > 0:
            self.ec_buffer[a].external_value[s] = max(self.ec_buffer[a].external_value[s],
                                                      (r + self.gamma * r_loop) / (1 - self.gamma ** d))
            d = 0
            r_loop = 0
        else:
            self.ec_buffer[a].external_value[s] = max(self.ec_buffer[a].external_value[s], r + self.gamma * r_e)
        self.ec_buffer[a].internal_value[s] = min(self.ec_buffer[a].internal_value[s], r_i)
        if self.ec_buffer[a].external_value[s] < old_r_i or self.ec_buffer[a].external_value[s] > old_r_e + 1e-7:
            r_i = max([buffer.internal_value[s] for buffer in self.ec_buffer])
            if d > 0:
                r_loop = r_loop * self.gamma + r
            for stm1, atm1 in self.ec_buffer[a].prev_id:
                self.reward_update(s0, stm1, atm1, r_i, self.ec_buffer[a].external_value[s], d + 1, r_loop)

    def update_sequence(self, sequence, gamma):
        Rtn = [0]
        index = []
        for s, a, r, sp, ap, done in reversed(sequence):
            Rtn.append(self.gamma * Rtn[-1] + r)
            # print(rtn)
            # self.table[s, a] = max(self.table[s, a], Rtn[-1])
            ind = self.ec_buffer[a].peek(s)
            if ind < 0:
                ind = self.add(a, s, Rtn[-1], r, done)
            self.ec_buffer[a].internal_value[ind] = min(self.ec_buffer[a].internal_value[ind], - done * self.rmax)
            index.append(ind)
            if (s, a) not in self.ec_buffer[ap].prev_id[sp]:
                self.ec_buffer[ap].prev_id[sp].append((s, a))

        Rtn.pop()
        for i in range(len(sequence)):
            rtn = Rtn.pop()
            _, a, r, _, done = sequence[i]
            s = index[i]
            self.reward_update(s, s, a, (1 - done) * self.rmax, rtn, 0, 0)
