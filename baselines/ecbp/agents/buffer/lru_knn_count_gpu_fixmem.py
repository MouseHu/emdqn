import numpy as np
from baselines.deepq.experiments.atari.knn_cuda_fixmem import knn as knn_cuda_fixmem
import logging


# each action -> a lru_knn buffer
# alpha is for updating the internal reward i.e. count based reward
class LRU_KNN_COUNT_GPU_FIXMEM(object):
    def __init__(self, capacity, z_dim, env_name, action, num_actions=6, knn=4, alpha=0.1, beta=0.01):
        self.action = action
        self.alpha = alpha
        self.beta = beta
        self.env_name = env_name
        self.capacity = capacity
        self.num_actions = num_actions
        self.states = np.empty((capacity, z_dim), dtype=np.float32)
        self.external_value = np.zeros(capacity)
        self.internal_value = np.zeros(capacity)
        self.rmax = self.beta * 7200
        self.count = np.zeros(capacity)
        self.lru = np.zeros(capacity)
        # self.best_action = np.zeros((capacity, num_actions), dtype=np.int)
        self.curr_capacity = 0
        self.tm = 0.0
        self.addnum = 0
        self.buildnum = 256
        self.buildnum_max = 256
        self.bufpath = './buffer/%s' % self.env_name
        self.threshold = 1e-8
        self.knn = knn
        self.logger = logging.getLogger("ecbp")
        # self.beta = beta
        self.address = knn_cuda_fixmem.allocate(capacity, z_dim, 32, knn)

    def peek(self, key, external_value, internal_value, modify=False):
        if self.curr_capacity == 0:
            return None, self.rmax
        # print(np.array(key).shape)
        key = np.array(key, copy=True).squeeze()
        if len(key.shape) == 1:
            key = key[np.newaxis, ...]
        dist, ind = knn_cuda_fixmem.knn(self.address, key, 1, int(self.curr_capacity))
        dist, ind = np.transpose(dist), np.transpose(ind - 1)
        ind = ind[0][0]
        # print(dist.shape,ind.shape)
        if dist[0][0] < self.threshold:
            # print("peek success")
            self.lru[ind] = self.tm
            self.tm += 0.01
            if modify:
                if external_value > self.external_value[ind]:
                    self.external_value[ind] = external_value
                self.count[ind] += 1
                self.internal_value[ind] = (1 - self.alpha) * self.internal_value[ind] + self.alpha * internal_value
            return self.external_value[ind], self.beta / np.sqrt(self.count[ind]) if self.count[ind] > 0 else self.rmax

        return None, self.rmax

    def log(self, *args, logtype='debug', sep=' '):
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    def knn_value(self, key, knn):
        # knn = min(self.curr_capacity, knn)
        if self.curr_capacity < knn:
            return self.beta, self.beta
        key = np.array(key, copy=True).squeeze()
        if len(key.shape) == 1:
            key = key[np.newaxis, ...]
        dist, ind = knn_cuda_fixmem.knn(self.address, key, knn, int(self.curr_capacity))
        dist, ind = np.transpose(dist), np.transpose(ind - 1)
        coeff = np.exp(dist[0])
        coeff = coeff / np.sum(coeff)
        value = 0.0
        count = 0.0
        # print("nearest dist", dist[0][0])
        for j, index in enumerate(ind[0]):
            value += (self.internal_value[index] + self.external_value[index]) * coeff[j]
            count += 1 * coeff[j]
            self.lru[index] = self.tm
            self.tm += 0.01

        return value, self.beta / np.sqrt(count)

    def act_value(self, key, knn):
        # knn = min(self.curr_capacity, knn)
        external_values = []
        internal_values = []
        exact_refer = []
        if self.curr_capacity < knn:
            for i in range(len(key)):
                external_values.append(0)
                internal_values.append(self.rmax)
                exact_refer.append(False)
            return external_values, internal_values, np.array(exact_refer)

        key = np.array(key, copy=True).squeeze()
        if len(key.shape) == 1:
            key = key[np.newaxis, ...]
        dist, ind = knn_cuda_fixmem.knn(self.address, key, knn, int(self.curr_capacity))
        dist, ind = np.transpose(dist), np.transpose(ind - 1)
        self.log("norm", np.linalg.norm(key.squeeze()))
        self.log("dist", dist)
        # print(dist.shape, ind.shape, len(key), key.shape)
        # print("nearest dist", dist[0][0])
        for i in range(len(dist)):
            external_value = 0
            coeff = np.exp(-dist[i])
            coeff = coeff / np.sum(coeff)
            if dist[i][0] < self.threshold:
                exact_refer.append(True)
                external_value = self.external_value[ind[i][0]]
                internal_value = self.internal_value[ind[i][0]]
                # count = self.count[ind[i][0]]
                self.lru[ind[i][0]] = self.tm
                self.tm += 0.01
            else:
                exact_refer.append(False)
                internal_value = self.rmax
                for j, index in enumerate(ind[i]):
                    external_value += (self.external_value[index]) * coeff[j]
                    # print(coeff.shape, index, i)
                    self.lru[index] = self.tm
                    self.tm += 0.01

            external_values.append(external_value)
            internal_values.append(internal_value)

        return external_values, internal_values, np.array(exact_refer)

    def add(self, key, external_value, internal_value):
        # print(np.array(key).shape)
        if self.curr_capacity >= self.capacity:
            # find the LRU entry
            old_index = np.argmin(self.lru)
            self.states[old_index] = key
            self.external_value[old_index] = external_value
            self.internal_value[old_index] = internal_value
            self.lru[old_index] = self.tm
            self.count[old_index] = 2
            knn_cuda_fixmem.add(self.address, int(old_index), np.array(key))

        else:
            self.states[self.curr_capacity] = key
            self.external_value[self.curr_capacity] = external_value
            self.internal_value[self.curr_capacity] = internal_value
            self.lru[self.curr_capacity] = self.tm
            self.count[self.curr_capacity] = 2
            knn_cuda_fixmem.add(self.address, int(self.curr_capacity), np.array(key))
            self.curr_capacity += 1
        self.tm += 0.01

    def update_kdtree(self):
        pass
