import numpy as np
import copy
import logging
from sklearn.neighbors import KDTree


# each action -> a lru_knn buffer
# alpha is for updating the internal reward i.e. count based reward
class LRU_KNN_PS(object):
    def __init__(self, capacity, obs_shape, z_dim, env_name, action, num_actions=6, knn=4, debug=True, gamma=0.99, alpha=0.1,
                 beta=0.01):
        self.obs = np.empty((capacity,) + obs_shape, dtype=np.uint8)
        self.action = action
        self.alpha = alpha
        self.beta = beta
        self.z_dim = z_dim
        self.env_name = env_name
        self.capacity = capacity
        self.num_actions = num_actions
        self.rmax = 100000
        self.states = np.empty((capacity, z_dim), dtype=np.float32)
        self.external_value = np.full((capacity, num_actions), np.nan)
        self.state_value_v = np.full((capacity,), np.nan)
        self.state_value_u = np.full((capacity,), np.nan)
        self.reward = np.zeros((capacity, num_actions))
        self.done = np.zeros((capacity, num_actions), dtype=np.bool)
        self.newly_added = np.ones((capacity, num_actions), dtype=np.bool)
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
        self.curr_capacity = 0
        self.tm = 0.0
        self.threshold = 1e-7
        self.knn = knn
        self.gamma = gamma
        self.b = 0.01
        self.knn = knn
        # self.beta = beta
        self.tree = None
        self.logger = logging.getLogger("ecbp")

    def log(self, *args, logtype='debug', sep=' '):
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    def build_tree(self):
        if self.curr_capacity == 0:
            return False
        self.tree = KDTree(self.states[:self.curr_capacity], leaf_size=10)
        return True

    def peek(self, key):
        if self.curr_capacity == 0 or self.tree is None:
            return -1, [], []
        # print(np.array(key).shape)
        key = np.array(key, copy=True)
        if len(key.shape) == 1:
            key = key[np.newaxis, ...]
        dist, ind = self.tree.query(key, k=min(self.knn, self.curr_capacity))
        # dist, ind = knn_cuda_fixmem.knn(self.address, key, 1, self.curr_capacity)
        # dist, ind = np.transpose(dist), np.transpose(ind - 1)
        ind_n = ind[0][0]
        if dist[0][0] < self.threshold:
            return ind_n, dist, ind
        return -1, dist, ind

    def act_value(self, key, knn):
        knn = min(self.curr_capacity, knn)

        internal_values = []
        external_values = []
        exact_refer = []
        if knn < 1 or self.tree is None:
            for i in range(len(key)):
                internal_values.append(self.rmax * np.ones(self.num_actions))
                external_values.append(np.zeros(self.num_actions))
                exact_refer.append(False)
            return external_values, internal_values, np.array(exact_refer)

        key = np.array(key, copy=True)

        if len(key.shape) == 1:
            key = key[np.newaxis, ...]
        assert key.shape[0] == 1
        dist, ind = self.tree.query(key, k=min(knn + 1, self.curr_capacity))
        # dist, ind = knn_cuda_fixmem.knn(self.address, key, knn, self.curr_capacity)
        # dist, ind = np.transpose(dist), np.transpose(ind - 1)
        # print(dist.shape, ind.shape, len(key), key.shape)
        # print("nearest dist", dist[0][0])
        external_value = np.zeros(self.num_actions)
        external_nan_mask = np.full((self.num_actions,), np.nan)
        internal_value = self.rmax * np.ones(self.num_actions)
        old_mask = np.array([[1 - self.newly_added[i] for i in query] for query in ind]).astype(np.bool)
        ind, dist = ind[old_mask].reshape(1, -1), dist[old_mask].reshape(1, -1)
        for i in range(len(dist)):
            coeff = -dist[i] / self.b
            coeff = coeff - np.max(coeff)
            coeff = np.exp(coeff)
            coeff = coeff / np.sum(coeff)
            if dist[i][0] < self.threshold and not np.isnan(self.external_value[ind[i][0]]).all():

                self.log("peek in act ", ind[i][0])
                exact_refer.append(True)
                external_value = copy.deepcopy(self.external_value[ind[i][0]])
                internal_value = copy.deepcopy(self.internal_value[ind[i][0]])
                # external_value[np.isnan(external_value)] = 0
                self.lru[ind[i][0]] = self.tm
                self.tm += 0.01
            else:
                exact_refer.append(False)
                for j, index in enumerate(ind[i]):
                    tmp_external_value = copy.deepcopy(self.external_value[index, :])
                    tmp_external_value[np.isnan(tmp_external_value)] = 0
                    external_nan_mask[(1 - np.isnan(tmp_external_value)).astype(np.bool)] = 0
                    external_value += tmp_external_value * coeff[j]
                    self.lru[index] = self.tm
                    self.tm += 0.01
                external_value += external_nan_mask
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
        if self.internal_value[src, action] > 0 and sum(self.next_id[src][action].values()) > 5:
            self.internal_value[src, action] = 0
        self.reward[src, action] = reward  # note that we assume that reward function is deterministic
        self.done[src, action] = done
        return sum(self.next_id[src][action].values())

    def add_node(self, key,obs=None):
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
            self.external_value[old_index] = np.full((self.num_actions,), np.nan)
            self.internal_value[old_index] = self.rmax * np.ones(self.num_actions)
            self.state_value_u[old_index] = np.nan
            self.state_value_v[old_index] = np.nan
            self.lru[old_index] = self.tm
            self.count[old_index] = 2
            if obs is not None:
                self.obs[old_index] = obs
            self.prev_id[old_index] = []
            # knn_cuda_fixmem.add(self.address, old_index, np.array(key))
            self.tm += 0.01
            # self.build_tree()
            return old_index, True

        else:
            self.states[self.curr_capacity] = key
            self.lru[self.curr_capacity] = self.tm
            self.count[self.curr_capacity] = 2
            if obs is not None:
                self.obs[self.curr_capacity] = obs
            # knn_cuda_fixmem.add(self.address, self.curr_capacity, np.array(key))
            self.curr_capacity += 1
            self.tm += 0.01
            # self.build_tree()
            return self.curr_capacity - 1, False

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
        sample_size = min(self.curr_capacity, sample_size)
        if sample_size % 2 == 1:
            sample_size -= 1
        if sample_size < 2:
            return None
        indexes = []
        positives = []
        values = []
        actions = []
        while len(indexes) < sample_size:
            ind = int(np.random.randint(0, self.curr_capacity, 1))
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
        z_target = [self.states[ind] for ind in indexes]
        z_pos = [self.states[pos] for pos in positives]
        z_neg = [self.states[neg] for neg in negatives]
        return indexes, positives, negatives, z_target, z_pos, z_neg, values, actions

    def update(self, indexes, z_new):
        self.log("update in buffer", self.curr_capacity)
        assert len(indexes) == len(z_new), "{}{}".format(len(indexes), len(z_new))
        assert z_new.shape[1] == self.z_dim
        for i, ind in enumerate(indexes):
            self.states[ind] = z_new[i]
