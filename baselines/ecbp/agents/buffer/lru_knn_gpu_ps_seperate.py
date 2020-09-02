import numpy as np
from baselines.deepq.experiments.atari.knn_cuda_fixmem import knn as knn_cuda_fixmem
import copy
import logging


# each action -> a lru_knn buffer
# alpha is for updating the internal reward i.e. count based reward
class LRU_KNN_GPU_PS_SEPERATE(object):
    def __init__(self, capacity, z_dim, env_name, action, num_actions=6, knn=4, debug=True, gamma=0.99,
                 alpha=0.1,
                 beta=0.01):
        self.action = action
        self.alpha = alpha
        self.beta = beta
        self.env_name = env_name
        self.capacity = capacity
        self.num_actions = num_actions
        self.rmax = 100000
        self.states = np.empty((capacity, z_dim), dtype=np.float32)
        self.address = np.full((capacity, num_actions), np.nan)
        self.reversed_address = np.full((capacity, num_actions), np.nan)
        # self.hash_table = np.empty((capacity, z_dim), dtype=np.float32)
        # self.hashes = {}
        self.external_value = np.full((capacity, num_actions), np.nan)
        self.state_value_v = np.full((capacity,), np.nan)
        self.state_value_u = np.full((capacity,), np.nan)
        self.reward = np.zeros((capacity, num_actions))
        self.done = np.zeros((capacity, num_actions), dtype=np.bool)
        self.newly_added = np.ones((capacity,), dtype=np.bool)
        self.internal_value = self.rmax * np.ones((capacity, num_actions))
        self.prev_id = [[] for _ in range(capacity)]
        self.next_id = [[{} for __ in range(num_actions)] for _ in range(capacity)]
        self.pseudo_count = [[{} for __ in range(num_actions)] for _ in range(capacity)]
        self.pseudo_reward = np.zeros((capacity, num_actions))
        self.pseudo_prev = [{} for _ in range(capacity)]
        self.debug = debug
        self.count = np.zeros((capacity, num_actions))
        self.lru = np.zeros(capacity)
        # self.best_action = np.zeros((capacity, num_actions), dtype=np.int)
        # self.gpu_capacity[0] = 0
        self.tm = 0.0
        self.threshold = 1e-7
        self.knn = knn
        self.gamma = gamma
        self.b = 0.01
        self.z_dim = z_dim
        # self.beta = beta
        batch_size = 32
        self.gpu_address = [knn_cuda_fixmem.allocate(capacity, z_dim, batch_size, knn+1) for _ in range(num_actions+1)]
        self.gpu_capacity = [0 for _ in range(num_actions+1)]
        self.logger = logging.getLogger("ecbp")

    def log(self, *args, logtype='debug', sep=' '):
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    def peek(self, key):
        if self.gpu_capacity[0] == 0:
            return -1, [], []
        key = np.array(key, copy=True).squeeze()
        key_norm= np.linalg.norm(key)
        if len(key.shape) == 1:
            key = key[np.newaxis, ...]
        dist, ind = knn_cuda_fixmem.knn(self.address, key, min(self.knn, self.gpu_capacity[0]), int(self.gpu_capacity[0]))
        dist, ind = np.transpose(dist), np.transpose(ind - 1)
        ind_n = ind[0][0]
        if dist[0][0] < self.threshold*key_norm:
            return ind_n, dist, ind
        return -1, dist, ind

    def act_value(self, key, knn):
        pass

    def add_edge(self, src, des, action, reward, done):
        if (src, action) not in self.prev_id[des]:
            self.prev_id[des].append((src, action))
            # self.newly_added[src, action] = True
        try:
            self.next_id[src][action][des] += 1
        except KeyError:
            self.next_id[src][action][des] = 1
        except IndexError:
            print(len(self.next_id))
            print(len(self.next_id[src]))
            print(self.next_id[src][action])
            raise IndexError
        if self.internal_value[src, action] > 0 and sum(self.next_id[src][action].values()) > 5:
            self.internal_value[src, action] = 0
        self.reward[src, action] = reward  # note that we assume that reward function is deterministic
        self.done[src, action] = done
        return sum(self.next_id[src][action].values())

    def add_node(self, key):
        # print(np.array(key).shape)
        if self.gpu_capacity[0] >= self.capacity:
            # find the LRU entry
            old_index = int(np.argmin(self.lru))
            for action in range(self.num_actions):
                for successor in self.next_id[old_index][action].keys():
                    for s, a in self.prev_id[successor]:
                        if s == old_index:
                            self.prev_id[successor].remove((s, a))
                self.next_id[old_index][action] = dict()
            self.states[old_index] = key
            self.external_value[old_index] = np.full((self.num_actions,), np.nan)
            self.internal_value[old_index] = self.rmax * np.ones(self.num_actions)
            self.state_value_u[old_index] = np.nan
            self.state_value_v[old_index] = np.nan
            self.lru[old_index] = self.tm
            self.count[old_index] = 2
            self.prev_id[old_index] = []
            knn_cuda_fixmem.add(self.address, int(old_index), np.array(key))
            self.tm += 0.01
            self.newly_added[old_index] = True
            return old_index, True

        else:
            self.states[self.gpu_capacity[0]] = key
            self.lru[self.gpu_capacity[0]] = self.tm
            self.newly_added[self.gpu_capacity[0]] = True
            self.count[self.gpu_capacity[0]] = 2
            knn_cuda_fixmem.add(self.address, int(self.gpu_capacity[0]), np.array(key))
            self.gpu_capacity[0] += 1
            self.tm += 0.01

            return self.gpu_capacity[0] - 1, False

    @staticmethod
    def distance(a, b):
        return np.sqrt(np.sum(np.square(a - b)))

    def update_q_value(self, state, action, state_tp1, delta_u):
        successor_states = self.next_id[state][action].keys()
        weight = {s: self.next_id[state][action][s] for s in successor_states}
        trans_p = weight[state_tp1] / sum(weight.values())
        assert 0 <= trans_p <= 1
        if np.isnan(self.external_value[state, action]):
            self.external_value[state, action] = self.reward[state, action]
        self.external_value[state, action] += self.gamma * trans_p * delta_u

    def sample(self, sample_size):
        sample_size = min(self.gpu_capacity[0], sample_size)
        if sample_size % 2 == 1:
            sample_size -= 1
        if sample_size < 2:
            return None
        indexes = []
        positives = []
        values = []
        actions = []
        while len(indexes) < sample_size:
            ind = int(np.random.randint(0, self.gpu_capacity[0], 1))
            if ind in indexes:
                continue
            next_id_tmp = [[(a, ind_tp1) for ind_tp1 in self.next_id[ind][a].keys()] for a in range(self.num_actions)]
            next_id = []
            for x in next_id_tmp:
                next_id += x
            # next_id = np.array(next_id).reshape(-1)
            if len(next_id) == 0:
                continue
            positive = next_id[np.random.randint(0, len(next_id))][1]
            action = next_id[np.random.randint(0, len(next_id))][0]
            indexes.append(ind)
            positives.append(positive)
            actions.append(action)
            values.append(np.nanmax(self.external_value[ind, :]))

        negatives = [int((pos + sample_size // 2) % sample_size) for pos in positives]
        neighbours_index = self.knn_index(indexes)

        neighbours_value = np.array(
            [[np.nanmax(self.external_value[ind, :]) for ind in inds] for inds in neighbours_index])
        neighbours_index = np.array(neighbours_index).reshape(-1)
        # z_target = [self.states[ind] for ind in indexes]
        # z_pos = [self.states[pos] for pos in positives]
        # z_neg = [self.states[neg] for neg in negatives]
        return indexes, positives, negatives, values, actions, neighbours_index, neighbours_value

    def knn_index(self, index):
        dist, ind = knn_cuda_fixmem.knn(self.address, self.states[index], min(self.knn+1, self.gpu_capacity[0]),
                                        int(self.gpu_capacity[0]))
        dist, ind = np.transpose(dist), np.transpose(ind - 1)
        ind = ind[:,:-1]
        return ind

    def update(self, indexes, z_new):
        self.log("update in buffer", self.gpu_capacity[0])
        assert len(indexes) == len(z_new), "{}{}".format(len(indexes), len(z_new))
        assert z_new.shape[1] == self.z_dim
        for i, ind in enumerate(indexes):
            self.states[ind] = z_new[i]
            knn_cuda_fixmem.add(self.address, int(ind), np.array(z_new[i]).squeeze())
