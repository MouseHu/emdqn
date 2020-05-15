import numpy as np
from baselines.deepq.experiments.atari.knn_cuda_fixmem import knn as knn_cuda_fixmem
import copy
import logging


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
        self.pseudo_count = [[{} for __ in range(num_actions)] for _ in range(capacity)]
        self.pseudo_reward = np.zeros((capacity, num_actions))
        self.debug = debug
        self.count = np.zeros((capacity, num_actions))
        self.lru = np.zeros(capacity)
        # self.best_action = np.zeros((capacity, num_actions), dtype=np.int)
        self.curr_capacity = 0
        self.tm = 0.0
        self.threshold = 1e-2
        self.knn = knn
        self.gamma = gamma
        self.b = 10
        # self.beta = beta
        self.address = knn_cuda_fixmem.allocate(capacity, z_dim, 32, knn)
        self.logger = logging.getLogger("ecbp")

    def log(self, *args, logtype='debug', sep=' '):
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    def peek(self, key):
        if self.curr_capacity == 0:
            return -1, [], []
        # print(np.array(key).shape)
        key = np.array(key, copy=True)
        if len(key.shape) == 1:
            key = key[np.newaxis, ...]
        dist, ind = knn_cuda_fixmem.knn(self.address, key, 1, self.curr_capacity)
        dist, ind = np.transpose(dist), np.transpose(ind - 1)
        ind_n = ind[0][0]
        if dist[0][0] < self.threshold:
            return ind_n, dist, ind
        return -1, dist, ind

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
        external_value = np.zeros(self.num_actions)
        internal_value = self.rmax * np.ones(self.num_actions)
        for i in range(len(dist)):
            coeff = np.exp(dist[i] / self.b)
            coeff = coeff / np.sum(coeff)
            if dist[i][0] < self.threshold:

                self.log("peek in act ", ind[i][0])
                exact_refer.append(True)
                external_value = copy.deepcopy(self.external_value[ind[i][0]])
                internal_value = copy.deepcopy(self.internal_value[ind[i][0]])
                self.lru[ind[i][0]] = self.tm
                self.tm += 0.01
            else:
                exact_refer.append(False)
                for j, index in enumerate(ind[i]):
                    external_value += self.external_value[index] * coeff[j]
                    self.lru[index] = self.tm
                    self.tm += 0.01
            external_values.append(external_value)
            internal_values.append(internal_value)

        return external_values, internal_values, np.array(exact_refer)

    def add_edge(self, src, des, action, reward, done):
        if (src, action) not in self.prev_id[des]:
            self.prev_id[des].append((src, action))
            self.newly_added[src, action] = True
        try:
            self.next_id[src][action][des] += 1
        except KeyError:
            self.next_id[src][action][des] = 1
        if self.internal_value[src, action] > 0 and sum(self.next_id[src][action].values()) > 10:
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
            return old_index, True

        else:
            self.states[self.curr_capacity] = key
            self.lru[self.curr_capacity] = self.tm
            self.count[self.curr_capacity] = 2
            knn_cuda_fixmem.add(self.address, self.curr_capacity, np.array(key))
            self.curr_capacity += 1
            self.tm += 0.01
            return self.curr_capacity - 1, False

    @staticmethod
    def distance(a, b):
        return np.sqrt(np.sum(np.square(a - b)))

    def update_q_value(self, state, action, state_tp1, delta_u, kernel=False):
        successor_states = self.next_id[state][action].keys()
        if kernel:
            weight = {
                s: self.next_id[state][action][s] * np.exp(self.distance(self.states[s], self.states[state]) / self.b)
                for s in successor_states}
        else:
            weight = {s: self.next_id[state][action][s] for s in successor_states}
        trans_p = weight[state_tp1] / sum(weight.values())
        assert 0 <= trans_p <= 1
        self.external_value[state, action] += self.gamma * trans_p * delta_u
