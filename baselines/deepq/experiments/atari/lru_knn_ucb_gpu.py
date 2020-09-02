import numpy as np
from baselines.deepq.experiments.atari.knn_cuda import knn as knn_cuda


# each action -> a lru_knn buffer
class LRU_KNN_UCB_GPU(object):
    def __init__(self, capacity, z_dim, env_name, action, mode="mean", num_actions=6,knn=4):
        self.action = action
        self.knn = knn
        self.env_name = env_name
        self.capacity = capacity
        self.num_actions = num_actions
        self.states = np.empty((capacity, z_dim), dtype=np.float32)
        self.q_values_decay = np.zeros(capacity)
        self.count = np.zeros(capacity)
        self.lru = np.zeros(capacity)
        self.best_action = np.zeros((capacity, num_actions), dtype=np.int)
        self.curr_capacity = 0
        self.tm = 0.0
        self.addnum = 0
        self.buildnum = 256
        self.buildnum_max = 256
        self.bufpath = './buffer/%s' % self.env_name
        self.mode = mode
        self.threshold = 1e-2

    def peek(self, key, value_decay, action=-1, modify=False):
        if self.curr_capacity ==0 :
            return None, None, None

        dist, ind = knn_cuda.knn(np.transpose(np.array([key])), np.transpose(self.states[:self.curr_capacity]), 1)
        dist, ind = np.transpose(dist), np.transpose(ind - 1)
        ind = ind[0][0]
        # print(dist.shape,ind.shape)
        if dist[0][0] < self.threshold:
            # print("peek success")
            self.lru[ind] = self.tm
            self.tm += 0.01
            if modify:
                if self.mode == "max":
                    if value_decay > self.q_values_decay[ind]:
                        self.q_values_decay[ind] = value_decay
                        if action >= 0:
                            self.best_action[ind, action] = 1
                elif self.mode == "mean":
                    self.q_values_decay[ind] = (value_decay + self.q_values_decay[ind] * self.count[ind]) / (
                            self.count[ind] + 1)
                self.count[ind] += 1
            return self.q_values_decay[ind], self.best_action[ind], self.count[ind]
        # print self.states[ind], key
        # if prints:
        #     print("peek", dist[0][0])
        return None, None, None

    def knn_value(self, key, knn, ):
        # knn = min(self.curr_capacity, knn)
        if self.curr_capacity < knn:
            return 0.0, None, 1.0

        dist, ind = knn_cuda.knn(np.transpose(key), np.transpose(self.states[:self.curr_capacity]), knn)
        dist, ind = np.transpose(dist), np.transpose(ind - 1)
        coeff = np.exp(dist[0])
        coeff = coeff / np.sum(coeff)
        value = 0.0
        action = np.zeros((self.num_actions,))
        value_decay = 0.0
        count = 0
        # print("nearest dist", dist[0][0])
        for j, index in enumerate(ind[0]):
            value_decay += self.q_values_decay[index] * coeff[j]
            count += self.count[index] * coeff[j]
            action += self.best_action[index] * coeff[j]
            self.lru[index] = self.tm
            self.tm += 0.01

        q_decay = value_decay

        return q_decay, action, count

    def act_value(self, key, knn):
        # knn = min(self.curr_capacity, knn)
        values = []
        actions = np.zeros((len(key), self.num_actions))
        counts = []
        exact_refer = []
        if self.curr_capacity < knn:
            for i in range(len(key)):
                actions[i, self.action] = 1
                values.append(0)
                counts.append(1)
                exact_refer.append(False)
            return values, actions, counts, np.array(exact_refer)

        dist, ind = knn_cuda.knn(np.transpose(key), np.transpose(self.states[:self.curr_capacity]), knn)
        dist, ind = np.transpose(dist), np.transpose(ind - 1)
        # print(dist.shape, ind.shape, len(key), key.shape)
        # print("nearest dist", dist[0][0])
        for i in range(len(dist)):
            value_decay = 0
            count = 0
            coeff = np.exp(dist[i])
            coeff = coeff / np.sum(coeff)
            if dist[i][0] < self.threshold:
                exact_refer.append(True)
                value_decay = self.q_values_decay[ind[i][0]]
                count = self.count[ind[i][0]]
                actions[i] = self.best_action[ind[i][0]]
                self.lru[ind[i][0]] = self.tm
                self.tm += 0.01
            else:
                exact_refer.append(False)
                for j, index in enumerate(ind[i]):
                    value_decay += self.q_values_decay[index] * coeff[j]
                    count += self.count[index] * coeff[j]
                    # print(coeff.shape, index, i)
                    actions[i] += self.best_action[index] * coeff[j]
                    self.lru[index] = self.tm
                    self.tm += 0.01
            values.append(value_decay)
            counts.append(count)

        return values, actions, counts, np.array(exact_refer)

    def add(self, key, value_decay, action=-1):
        if self.curr_capacity >= self.capacity:
            # find the LRU entry
            old_index = np.argmin(self.lru)
            self.states[old_index] = key
            self.q_values_decay[old_index] = value_decay
            self.lru[old_index] = self.tm
            self.count[old_index] = 2
            if action >= 0:
                self.best_action[old_index, action] = 1
        else:
            self.states[self.curr_capacity] = key
            self.q_values_decay[self.curr_capacity] = value_decay
            self.lru[self.curr_capacity] = self.tm
            self.count[self.curr_capacity] = 2
            if action >= 0:
                self.best_action[self.curr_capacity, action] = 1
            self.curr_capacity += 1
        self.tm += 0.01

    def update_kdtree(self):
        pass
