import numpy as np
from baselines.deepq.experiments.atari.knn_cuda_fixmem import knn as knn_cuda_fixmem
import copy


# each action -> a lru_knn buffer
# alpha is for updating the internal reward i.e. count based reward
class LRU_KNN_GPU_PS(object):
    def __init__(self, capacity, z_dim, env_name, action, num_actions=6, knn=4, debug=True, gamma=0.99, alpha=0.1,
                 beta=0.01):
        self.action = action
        self.alpha = alpha
        self.beta = beta
        self.env_name = env_name
        self.capacity = capacity
        self.num_actions = num_actions
        self.rmax = 100000
        self.states = np.empty((capacity, z_dim), dtype=np.float32)
        self.external_value = np.zeros((capacity, num_actions))
        self.state_value_v = np.zeros(capacity)
        self.state_value_u = np.zeros(capacity)
        self.reward = np.zeros((capacity, num_actions))
        self.done = np.zeros((capacity, num_actions), dtype=np.bool)
        self.newly_added = np.ones((capacity, num_actions), dtype=np.bool)
        self.internal_value = self.rmax * np.ones((capacity, num_actions))
        self.prev_id = [[] for _ in range(capacity)]
        self.next_id = [[{} for __ in range(num_actions)] for _ in range(capacity)]
        self.debug = debug
        self.count = np.zeros((capacity, num_actions))
        self.lru = np.zeros(capacity)
        # self.best_action = np.zeros((capacity, num_actions), dtype=np.int)
        self.curr_capacity = 0
        self.tm = 0.0
        self.threshold = 1e-7
        self.knn = knn
        self.gamma = gamma
        # self.beta = beta
        self.address = knn_cuda_fixmem.allocate(capacity, z_dim, 32, knn)

    def peek(self, key):
        if self.curr_capacity == 0:
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
        return -1

    def act_value(self, key, knn):
        knn = min(self.curr_capacity, knn)
        internal_values = []
        external_values = []
        exact_refer = []
        if knn < 1:
            for i in range(len(key)):
                internal_values.append(np.zeros(self.num_actions))
                external_values.append(np.zeros(self.num_actions))
                exact_refer.append(False)
            return external_values, internal_values, np.array(exact_refer)

        key = np.array(key, copy=True)
        if len(key.shape) == 1:
            key = key[np.newaxis, ...]
        dist, ind = knn_cuda_fixmem.knn(self.address, key, knn, self.curr_capacity)
        dist, ind = np.transpose(dist), np.transpose(ind - 1)
        # print(dist.shape, ind.shape, len(key), key.shape)
        # print("nearest dist", dist[0][0])
        external_values = np.zeros(self.num_actions)
        internal_values = np.zeros(self.num_actions)
        for i in range(len(dist)):
            coeff = np.exp(dist[i])
            coeff = coeff / np.sum(coeff)
            if dist[i][0] < self.threshold:
                if self.debug:
                    print(" ")
                    print("peek in act ", ind[i][0], flush=True)
                exact_refer.append(True)
                external_values = copy.deepcopy(self.external_value[ind[i][0]])
                internal_values = copy.deepcopy(self.internal_value[ind[i][0]])
                self.lru[ind[i][0]] = self.tm
                self.tm += 0.01
                break
            else:
                exact_refer.append(False)
                for j, index in enumerate(ind[i]):
                    self.lru[index] = self.tm
                    self.tm += 0.01

        return external_values, internal_values, np.array(exact_refer)

    def add_edge(self, src, des, action, reward, done):
        if (src, action) not in self.prev_id[des]:
            self.prev_id[des].append((src, action))
            self.newly_added[src, action] = True
        try:
            self.next_id[src][action][des] += 1
        except KeyError:
            self.next_id[src][action][des] = 1
        if self.internal_value[src, action] > 0 and sum(self.next_id[src][action].values()) > 1:
            self.internal_value[src, action] = 0
        self.reward[src, action] = reward  # note that we assume that reward function is deterministic
        self.done[src, action] = done
        return sum(self.next_id[src][action].values())

    def add_node(self, key):
        # print(np.array(key).shape)
        if self.curr_capacity >= self.capacity:
            # find the LRU entry
            old_index = int(np.argmin(self.lru))
            for action in range(self.num_actions):
                for successor in self.next_id[old_index][action].keys():
                    for s, a in self.prev_id[successor]:
                        if s == old_index:
                            self.prev_id.remove((s, a))
                self.next_id[old_index][action] = dict()
            self.states[old_index] = key
            self.external_value[old_index] = np.zeros(self.num_actions)
            self.internal_value[old_index] = self.rmax * np.ones(self.num_actions)
            self.state_value_u[old_index] = 0
            self.state_value_v[old_index] = 0
            self.lru[old_index] = self.tm
            self.count[old_index] = 2
            self.prev_id[old_index] = []
            knn_cuda_fixmem.add(self.address, old_index, np.array(key))
            self.tm += 0.01
            return old_index

        else:
            self.states[self.curr_capacity] = key
            self.lru[self.curr_capacity] = self.tm
            self.count[self.curr_capacity] = 2
            knn_cuda_fixmem.add(self.address, self.curr_capacity, np.array(key))
            self.curr_capacity += 1
            self.tm += 0.01
            return self.curr_capacity - 1

    def update_q_value(self, state, action, state_tp1, delta_u):
        trans_p = self.next_id[state][action][state_tp1] / sum(self.next_id[state][action].values())
        assert 0 <= trans_p <= 1
        self.external_value[state, action] += self.gamma * trans_p * delta_u
