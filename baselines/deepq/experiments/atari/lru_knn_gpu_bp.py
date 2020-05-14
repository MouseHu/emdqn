import numpy as np
from baselines.deepq.experiments.atari.knn_cuda_fixmem import knn as knn_cuda_fixmem


# each action -> a lru_knn buffer
# alpha is for updating the internal reward i.e. count based reward
class LRU_KNN_GPU_BP(object):
    def __init__(self, capacity, z_dim, env_name, action, num_actions=6, knn=4, alpha=0.1, beta=0.01):
        self.action = action
        self.alpha = alpha
        self.beta = beta
        self.env_name = env_name
        self.capacity = capacity
        self.num_actions = num_actions
        self.states = np.empty((capacity, z_dim), dtype=np.float32)
        self.external_value = np.zeros(capacity)
        self.reward = np.zeros(capacity)
        self.done = np.zeros(capacity, dtype=np.bool)
        self.internal_value = np.zeros(capacity)
        self.next_id = -1 * np.ones((capacity, 2))
        self.newly_added = np.ones(capacity, dtype=np.bool)
        self.prev_id = [[] for _ in range(capacity)]
        self.brothers = [[] for _ in range(capacity)]
        self.rmax = self.beta * 2400
        self.count = np.zeros(capacity)
        self.lru = np.zeros(capacity)
        # self.best_action = np.zeros((capacity, num_actions), dtype=np.int)
        self.curr_capacity = 0
        self.tm = 0.0
        self.threshold = 1e-7
        self.knn = knn
        # self.beta = beta
        self.address = knn_cuda_fixmem.allocate(capacity, z_dim, 32, knn)

    def peek(self, key):
        if self.curr_capacity == 0:
            # print("zero capacity")
            return -1
        # print(np.array(key).shape)
        key = np.array(key, copy=True)
        if len(key.shape) == 1:
            key = key[np.newaxis, ...]
        dist, ind = knn_cuda_fixmem.knn(self.address, key, 1, self.curr_capacity)
        dist, ind = np.transpose(dist), np.transpose(ind - 1)
        ind = ind[0][0]
        if dist[0][0] < self.threshold:
            return ind
        # print("large threshold!", dist[0][0])
        return -1

    def knn_value(self, key, knn):
        # knn = min(self.curr_capacity, knn)
        if self.curr_capacity < knn:
            return self.beta, self.beta
        key = np.array(key, copy=True)
        if len(key.shape) == 1:
            key = key[np.newaxis, ...]
        dist, ind = knn_cuda_fixmem.knn(self.address, key, knn, self.curr_capacity)
        dist, ind = np.transpose(dist), np.transpose(ind - 1)
        coeff = np.exp(dist[0])
        coeff = coeff / np.sum(coeff)
        external_value = 0.0
        internal_value = 0.0
        # print("nearest dist", dist[0][0])
        for j, index in enumerate(ind[0]):
            external_value += (self.internal_value[index] + self.external_value[index]) * coeff[j]
            internal_value += (self.internal_value[index] + self.external_value[index]) * coeff[j]
            self.lru[index] = self.tm
            self.tm += 0.01

        return external_value, internal_value

    def act_value(self, key, knn, bp=True):
        knn = min(self.curr_capacity, knn)
        internal_values = []
        external_values = []
        exact_refer = []
        if self.curr_capacity < 1:
            # print(self.curr_capacity, knn)
            for i in range(len(key)):
                internal_values.append(0)
                external_values.append(0)
                exact_refer.append(False)
            return external_values, internal_values, np.array(exact_refer)

        key = np.array(key, copy=True)
        if len(key.shape) == 1:
            key = key[np.newaxis, ...]
        dist, ind = knn_cuda_fixmem.knn(self.address, key, knn, self.curr_capacity)
        dist, ind = np.transpose(dist), np.transpose(ind - 1)
        # print(dist.shape, ind.shape, len(key), key.shape)
        # print("nearest dist", dist[0][0])
        for i in range(len(dist)):
            external_value = 0
            internal_value = 0
            coeff = np.exp(dist[i])
            coeff = coeff / np.sum(coeff)
            if dist[i][0] < self.threshold:
                exact_refer.append(True)
                external_value = self.external_value[ind[i][0]]
                internal_value = self.internal_value[ind[i][0]]
                self.lru[ind[i][0]] = self.tm
                self.tm += 0.01
                print(ind[i][0], end=" ", flush=True)
            else:
                print("dist", dist[i][0], end=" ", flush=True)
                exact_refer.append(False)
                for j, index in enumerate(ind[i]):
                    if not bp:
                        external_value += (self.external_value[index]) * coeff[j]

                    # print(coeff.shape, index, i)
                    self.lru[index] = self.tm
                    self.tm += 0.01
                # external_value += (self.external_value[index]) * coeff[j]
            external_values.append(external_value)
            internal_values.append(internal_value)
        # print(external_values, internal_values, np.array(exact_refer))
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
            knn_cuda_fixmem.add(self.address, old_index, np.array(key))

        else:
            self.states[self.curr_capacity] = key
            self.external_value[self.curr_capacity] = external_value
            self.internal_value[self.curr_capacity] = internal_value
            self.lru[self.curr_capacity] = self.tm
            self.count[self.curr_capacity] = 2
            knn_cuda_fixmem.add(self.address, self.curr_capacity, np.array(key))
            self.curr_capacity += 1
        self.tm += 0.01

    def update_kdtree(self):
        pass
