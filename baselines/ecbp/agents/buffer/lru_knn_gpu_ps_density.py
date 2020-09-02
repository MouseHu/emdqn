import numpy as np
from baselines.deepq.experiments.atari.knn_cuda_fixmem import knn as knn_cuda_fixmem
import copy
import logging


# each action -> a lru_knn buffer
# alpha is for updating the internal reward i.e. count based reward
class LRU_KNN_GPU_PS_DENSITY(object):
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
        # self.hash_table = np.empty((capacity, z_dim), dtype=np.float32)
        # self.hashes = {}
        self.knn_mean_dist = np.full((capacity,), 0)
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
        self.density = self.rmax * np.ones(capacity)
        self.neighbour_forward = [[] for _ in range(capacity)]
        self.neighbour_dist_forward = [[] for _ in range(capacity)]
        self.neighbour_backward = [[] for _ in range(capacity)]
        # self.best_action = np.zeros((capacity, num_actions), dtype=np.int)
        self.curr_capacity = 0
        self.tm = 0.0

        self.threshold = 1e-4

        self.knn = knn
        self.gamma = gamma
        self.b = 1
        self.z_dim = z_dim
        # self.beta = beta
        batch_size = 32

        self.batch_size = batch_size

        self.address = knn_cuda_fixmem.allocate(capacity, z_dim, batch_size, knn * max(self.num_actions, 4))
        self.logger = logging.getLogger("ecbp")

    def log(self, *args, logtype='debug', sep=' '):
        getattr(self.logger, logtype)(sep.join(str(a) for a in args))

    def peek(self, key):
        if self.curr_capacity == 0:
            return -1, [], []
        # print(np.array(key).shape)
        key = np.array(key, copy=True).squeeze()
        key_norm = np.linalg.norm(key)
        if len(key.shape) == 1:
            key = key[np.newaxis, ...]
        # self.log("begin knn",self.knn,self.curr_capacity,self.address,key.shape)
        dist, ind = knn_cuda_fixmem.knn(self.address, key, min(self.knn * 4, self.curr_capacity),
                                        int(self.curr_capacity))
        # dist, ind = knn_cuda_fixmem.knn(self.address, key, 1, self.curr_capacity)
        # self.log("finish knn")
        dist, ind = np.transpose(dist), np.transpose(ind - 1)
        ind_n = ind[0][0]
        # self.log("key_norm in peek", key_norm)
        if dist[0][0] < self.threshold * key_norm:
            # if ind_n != ind_hash:
            #     self.log("hash not found", ind_hash)
            return ind_n, dist, ind
        # if ind_n == -1:
        # self.log("pick exact failed. dist", dist[0][0], "z", key, "ind", ind_n)
        # if -1 != ind_hash and dist[0][0] > self.threshold:
        #     self.log("knn not found", ind_hash)
        return -1, dist, ind

    def act_value(self, key, knn):
        knn = min(self.curr_capacity, knn)
        internal_values = []
        external_values = []
        exact_refer = []
        if knn < 1:
            self.log("knn too small", logtype='info')
            for i in range(len(key)):
                internal_values.append(self.rmax * np.ones(self.num_actions))
                external_values.append(np.zeros(self.num_actions))
                exact_refer.append(False)
            return external_values, internal_values, np.array(exact_refer)

        key = np.array(key, copy=True).squeeze()
        key_norm = np.linalg.norm(key, ord=2)
        self.log("key_norm in act value", key_norm)
        if len(key.shape) == 1:
            key = key[np.newaxis, ...]

        # dist, ind = knn_cuda_fixmem.knn(self.address, key, knn, int(self.curr_capacity))
        dist, ind = knn_cuda_fixmem.knn_conditional(self.address, key, copy.copy(self.newly_added), knn,
                                                    int(self.curr_capacity))

        dist, ind = np.transpose(dist), np.transpose(ind - 1)
        # print(dist.shape, ind.shape, len(key), key.shape)
        # print("nearest dist", dist[0][0])
        external_value = np.zeros(self.num_actions)
        external_nan_mask = np.full((self.num_actions,), np.nan)
        internal_value = self.rmax * np.ones(self.num_actions)
        old_mask = np.array([[1 - self.newly_added[i] for i in query] for query in ind]).astype(np.bool)
        ind_new, dist_new = ind[old_mask], dist[old_mask]
        if len(dist_new) == 0:
            self.log("no old node", logtype='info')
            internal_values.append(self.rmax * np.ones(self.num_actions))
            external_values.append(np.zeros(self.num_actions))
            exact_refer.append(False)
            return external_values, internal_values, np.array(exact_refer)
        ind, dist = ind_new.reshape(1, -1), dist_new.reshape(1, -1)
        for i in range(len(dist)):
            self.log("compute coeff", dist, ind, len(dist), dist.shape)
            coeff = -dist[i] / self.b
            coeff = coeff - np.max(coeff)
            coeff = np.exp(coeff)
            coeff = coeff / np.sum(coeff)
            if dist[i][0] < self.threshold * key_norm and not np.isnan(self.external_value[ind[i][0]]).all():

                self.log("peek in act ", ind[i][0])
                exact_refer.append(True)
                external_value = copy.deepcopy(self.external_value[ind[i][0]])
                internal_value = copy.deepcopy(self.internal_value[ind[i][0]])
                # external_value[np.isnan(external_value)] = 0
                self.lru[ind[i][0]] = self.tm
                self.tm += 0.01
            else:
                exact_refer.append(False)
                self.log("inexact refer", ind[i][0], dist[i][0])
                self.log("coeff", coeff)
                for j, index in enumerate(ind[i]):
                    tmp_external_value = copy.deepcopy(self.external_value[index, :])
                    self.log("temp external value", self.external_value[index, :])
                    tmp_external_value[np.isnan(tmp_external_value)] = 0
                    external_nan_mask[(1 - np.isnan(tmp_external_value)).astype(np.bool)] = 0
                    external_value += tmp_external_value * coeff[j]
                    self.lru[index] = self.tm
                    self.tm += 0.01
                external_value += external_nan_mask
            external_values.append(external_value)
            internal_values.append(internal_value)

        return external_values, internal_values, np.array(exact_refer)

    def act_value_ec(self, key, knn):
        knn = min(self.curr_capacity // self.num_actions, knn)
        key = np.array(key, copy=True).squeeze()
        exact_refer = [0 for _ in range(self.num_actions)]
        if len(key.shape) == 1:
            key = key[np.newaxis, ...]

        if knn < 1:
            self.log("knn too small", logtype='info')
            return [np.zeros(self.num_actions)], [self.rmax * np.ones(self.num_actions)], exact_refer

        dist, ind = knn_cuda_fixmem.knn(self.address, key, knn * self.num_actions, int(self.curr_capacity))
        dist, ind = np.transpose(dist), np.transpose(ind - 1)

        external_values = self.external_value[ind[0]]
        external_value = -self.rmax * np.ones((self.num_actions,))
        internal_value = self.rmax * np.ones((self.num_actions,))
        for a in range(self.num_actions):
            # self.log("a")
            external_values_column = external_values[~np.isnan(external_values[:, a]), a]
            external_values_dist = dist[0][np.where(~np.isnan(external_values[:, a]))[0]]
            if len(external_values_dist) == 0:
                # not finding any value
                continue
            elif external_values_dist[0] < self.threshold:
                # find
                external_value[a] = external_values_column[0]
                internal_value[a] = 0
                exact_refer[a] = True
            else:
                knn_a = min(len(external_values_dist), knn)

                coeff = -external_values_dist[:knn_a] / self.b
                coeff = coeff - np.max(coeff)
                coeff = np.exp(coeff)
                coeff = coeff / np.sum(coeff)
                external_value[a] = np.dot(external_values_column[:knn_a], coeff)
                self.log("knn_a", knn_a, a)
                self.log("column", external_values_column[:knn_a])
                self.log("dist", external_values_dist[:knn_a])
                self.log("coeff", coeff)

        return [external_value], [internal_value], exact_refer

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

    def compute_density(self, ind):
        knn_count = min(self.knn, len(self.neighbour_dist_forward[ind]))
        knn_dist = np.sort(self.neighbour_dist_forward[ind])[:knn_count]
        if knn_count < 1:
            self.density[ind] = self.rmax
        else:
            self.density[ind] = np.average(knn_dist)

    def add_node(self, key, knn_dist, knn_ind):
        knn_dist = np.array(knn_dist).reshape(-1).tolist()
        knn_ind = np.array(knn_ind).reshape(-1).tolist()
        # print(np.array(key).shape)
        if self.curr_capacity >= self.capacity:
            # find the LRU entry

            old_index = int(np.argmin(self.density + 2 * self.rmax * self.newly_added + self.lru * 0.1))
            self.log("density based cache index:", old_index, "density:", self.density[old_index])
            # old_index = np.random.randint(0,self.capacity)

            # deal with model
            for action in range(self.num_actions):
                for successor in self.next_id[old_index][action].keys():
                    for s, a in self.prev_id[successor]:
                        if s == old_index:
                            self.prev_id[successor].remove((s, a))
                self.next_id[old_index][action] = dict()
            self.prev_id[old_index] = []
            # deal with neighbour
            for neighbour in self.neighbour_backward[old_index]:
                place = np.where(np.array(self.neighbour_forward[neighbour]) == old_index)[0]
                if len(place) == 0:
                    print("backward", old_index, self.neighbour_backward[old_index])
                    print("forward", neighbour, self.neighbour_forward[neighbour])
                    raise SyntaxError

                assert len(place) == 1, place

                place = int(place)
                self.neighbour_forward[neighbour].pop(place)
                self.neighbour_dist_forward[neighbour].pop(place)
                # print(self.neighbour_dist_forward[neighbour])
                self.compute_density(neighbour)
            for neighbour in self.neighbour_forward[old_index]:

                self.neighbour_backward[neighbour].remove(old_index)


            # clean buffer and then treat it like a new one
            self.density[old_index] = self.rmax
            self.external_value[old_index] = np.full((self.num_actions,), np.nan)
            self.internal_value[old_index] = self.rmax * np.ones(self.num_actions)
            self.state_value_u[old_index] = np.nan
            self.state_value_v[old_index] = np.nan
            self.neighbour_forward[old_index] = []
            self.neighbour_dist_forward[old_index] = []
            self.neighbour_backward[old_index] = []
            new_index = old_index

        else:
            new_index = self.curr_capacity
            self.curr_capacity += 1

        if new_index in knn_ind:

            place = np.where(np.array(new_index) == knn_ind)[0]

            assert len(place) == 1, place
            place = int(place)
            knn_ind.pop(place)
            knn_dist.pop(place)
        self.neighbour_forward[new_index] = knn_ind
        self.neighbour_dist_forward[new_index] = knn_dist
        # print(self.neighbour_dist_forward[new_index])
        self.compute_density(new_index)
        # print(knn_ind,knn_dist)
        # for x in zip([knn_ind, knn_dist]):
        #     print(x)
        for ind, dist in zip(knn_ind, knn_dist):
            self.neighbour_backward[ind].append(new_index)
            if len(self.neighbour_forward[ind]) < self.knn * 4 and new_index not in self.neighbour_forward[ind]:
                self.neighbour_forward[ind].append(new_index)
                self.neighbour_dist_forward[ind].append(dist)
                # print(self.neighbour_dist_forward[ind])
                self.compute_density(ind)
                self.neighbour_backward[new_index].append(ind)
            else:
                dist_max = np.max(self.neighbour_dist_forward[ind])
                ind_max = int(np.argmax(self.neighbour_dist_forward[ind]))

                if dist_max > dist:
                    ind_tmp = self.neighbour_forward[ind][ind_max]
                    try:
                        self.neighbour_backward[ind_tmp].remove(ind)
                    except ValueError:

                        print(ind_tmp, self.neighbour_forward[ind])
                        print(ind, self.neighbour_backward[ind_tmp])

                        raise ValueError
                    self.neighbour_forward[ind][ind_max] = new_index
                    self.neighbour_dist_forward[ind][ind_max] = dist
                    # print(self.neighbour_dist_forward[ind])
                    self.compute_density(ind)
                    self.neighbour_backward[new_index].append(ind)

        self.states[new_index] = key
        self.lru[new_index] = self.tm
        self.newly_added[new_index] = True
        self.count[new_index] = 2
        knn_cuda_fixmem.add(self.address, int(new_index), np.array(key))
        self.tm += 0.01
        return new_index, False

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


    def sample(self, sample_size, neg_num=1):

        sample_size = min(self.curr_capacity, sample_size)
        if sample_size % 2 == 1:
            sample_size -= 1
        if sample_size < 2:
            return None
        indexes = []
        positives = []
        negatives = []
        values = []
        actions = []

        rewards = []

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

            reward = self.reward[ind, action]
            indexes.append(ind)
            positives.append(positive)
            actions.append(action)
            rewards.append(reward)
            values.append(np.nanmax(self.external_value[ind, :]))

        while len(negatives) < sample_size * neg_num:
            ind = indexes[len(negatives) // neg_num]

            neg_ind = int(np.random.randint(0, self.curr_capacity, 1))
            neg_ind_next = [[ind_tp1 for ind_tp1 in self.next_id[neg_ind][a].keys()] for a in range(self.num_actions)]
            ind_next = [[ind_tp1 for ind_tp1 in self.next_id[ind][a].keys()] for a in range(self.num_actions)]
            if ind in neg_ind_next or neg_ind in ind_next or neg_ind == ind:
                continue
            negatives.append(neg_ind)
        neighbours_index = self.knn_index(indexes)

        neighbours_value = np.array(
            [[np.max(self.external_value[ind, :]) for ind in inds] for inds in neighbours_index])
        for i in range(len(neighbours_value)):
            neighbours_value[i][np.isnan(neighbours_value[i])] = values[i]
        neighbours_index = np.array(neighbours_index).reshape(-1)
        # z_target = [self.states[ind] for ind in indexes]
        # z_pos = [self.states[pos] for pos in positives]
        # z_neg = [self.states[neg] for neg in negatives]

        return indexes, positives, negatives, rewards, values, actions, neighbours_index, neighbours_value


    def knn_index(self, index):
        assert self.knn + 1 < self.curr_capacity
        dist, ind = knn_cuda_fixmem.knn(self.address, self.states[index], self.knn + 1,
                                        int(self.curr_capacity))
        dist, ind = np.transpose(dist), np.transpose(ind - 1)
        ind = ind[:, 1:]
        return ind

    def update(self, indexes, z_new):
        self.log("update in buffer", self.curr_capacity)
        assert len(indexes) == len(z_new), "{}{}".format(len(indexes), len(z_new))
        assert z_new.shape[1] == self.z_dim
        for i, ind in enumerate(indexes):
            self.states[ind] = z_new[i]
            knn_cuda_fixmem.add(self.address, int(ind), np.array(z_new[i]).squeeze())


    def recompute_density(self):
        self.neighbour_backward = [[] for _ in range(self.capacity)]
        for i in range(int(np.ceil(self.curr_capacity / self.batch_size))):
            low = i * self.batch_size
            high = min(self.curr_capacity, (i + 1) * self.batch_size)
            _, knn_dist, knn_ind = self.peek(
                self.states[low:high])

            self.neighbour_dist_forward[
            i * self.batch_size:min(self.curr_capacity, (i + 1) * self.batch_size)] = knn_dist
            for j in range(low, high):
                knn_ind_j = knn_ind[j-low].tolist()
                knn_ind_j.pop(0)
                self.neighbour_forward[j] = knn_ind_j
                knn_dist_j = knn_dist[j-low].tolist()
                knn_dist_j.pop(0)
                self.neighbour_dist_forward[j] = knn_dist_j
                # if j in knn_ind:
                #     place = np.where(knn_ind[j]==j)[0]
                #     if len(place)>0:
                #         place = int(place)
                #         self.neighbour_forward[j].pop(place)
                #         self.neighbour_dist_forward[j].pop(place)
                for ind in self.neighbour_forward[j]:
                    self.neighbour_backward[ind].append(j)
                self.compute_density(j)

