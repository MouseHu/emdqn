import numpy as np
from sklearn.neighbors import BallTree, KDTree
import os

import gc


# each action -> a lru_knn buffer
# there are two "key"s, a learned one and a fixed one.
# the fixed one ensures that our method should be better than baseline, i.e. model free episodic control
class LRU_KNN_SingleAction(object):
    def __init__(self, capacity, hash_dim, env_name,action_number=0):
        self.env_name = env_name
        self.capacity = capacity
        # self.states = np.empty((capacity, z_dim), dtype=np.float32)  # learned keys
        self.prev_action = [[] for _ in range(capacity)]
        self.next_action = [[] for _ in range(capacity)]
        self.next_id = [[] for _ in range(capacity)]
        self.prev_id = [[] for _ in range(capacity)]
        self.hashes = np.empty((capacity, hash_dim), dtype=np.float32)  # fixed keys
        # self.hashes = dict()  # fixed keys
        self.q_values_decay = np.zeros(capacity)
        self.lru = np.zeros(capacity)
        self.curr_capacity = 0
        self.tm = 0.0
        # self.tree = None
        self.hash_tree = None
        # self.hash_tree = None
        self.addnum = 0
        self.buildnum = 256
        self.buildnum_max = 256
        self.bufpath = './buffer/%s' % self.env_name
        self.build_tree_times = 0
        self.build_tree = False
        self.action_number = action_number

    def peek(self, h, value_decay, modify, prev_id=-1, prev_action=-1):
        if self.curr_capacity == 0 or self.build_tree is False:
            return None, None

        dist, ind = self.hash_tree.query([h], k=1)
        ind = ind[0][0]
        # if self.states[ind] == key:
        # if np.allclose(self.states[ind], key):
        # if np.allclose(self.hashes[ind], h, atol=1e-08):
        if dist[0][0] < 1e-9:
            self.lru[ind] = self.tm
            self.tm += 0.01
            if modify:
                if value_decay > self.q_values_decay[ind]:
                    self.q_values_decay[ind] = value_decay
                    if prev_id >= 0:
                        self.prev_id[ind].append(prev_id)
                        self.prev_action[ind].append(prev_action)
            return self.q_values_decay[ind], ind
        # print self.states[ind], key
        else:
            return None, None

    @staticmethod
    def switch_first_half(obs, obs_next, batch_size):
        half_size = int(batch_size / 2)
        tmp = obs[:half_size, ...]
        obs[:half_size, ...] = obs_next[:half_size, ...]
        obs_next[:half_size, ...] = tmp
        return obs, obs_next

    def update(self, idxes, values):
        assert len(idxes) == len(values)
        for id, value in zip(idxes, values):
            self.hashes[id] = value

    def sample_neg_keys(self, avoids, batch_size):
        # sample negative keys
        assert batch_size + len(
            avoids) <= self.curr_capacity, "can't sample that much neg samples from episodic memory!"
        idxes = []
        while len(idxes) < batch_size:
            id = np.random.randint(0, self.curr_capacity)

            if (id not in idxes) and not (np.array([np.array_equal(self.hashes[id], x) for x in avoids]).any()):
                idxes.append(id)
        return idxes

    def act_value(self, key, h, knn=4):
        value, _ = self.peek(None, h, modify=False)
        if value is not None:
            return value, True
        else:
            # print(self.curr_capacity,knn)
            return self.knn_value(key, knn=knn), False

    def knn_value(self, key, knn):
        knn = min(self.curr_capacity, knn)
        if self.curr_capacity == 0 or self.build_tree == False:
            return 0.0
        key = np.squeeze(key)
        dist, ind = self.hash_tree.query([key], k=knn)

        coeff = np.exp(dist[0])
        coeff = coeff / np.sum(coeff)
        # value = 0.0
        value_decay = 0.0
        # count = 0
        for j, index in enumerate(ind[0]):
            value_decay += self.q_values_decay[index] * coeff[j]
            # count += self.count[index] * coeff[j]
            self.lru[index] = self.tm
            self.tm += 0.01

        q_decay = value_decay

        return q_decay

    # def add(self, hash, value_decay, prev_id=-1, prev_action=-1):
    #     if self.curr_capacity >= self.capacity:
    #         # find the LRU entry
    #         old_index = int(np.argmin(self.lru))
    #         self.hashes[old_index] = hash
    #         # self.hashes[tuple(hash)] = old_index
    #         # self.obses[old_index] = obs
    #         self.q_values_decay[old_index] = value_decay
    #         self.lru[old_index] = self.tm
    #         if prev_id >= 0:
    #             self.next_id[prev_id] = [old_index]
    #             self.next_action[prev_id] = [self.action_number]
    #         self.prev_id[old_index] = [prev_id]
    #         self.prev_action[old_index] = [prev_action]
    #         # self.next_id[old_index] = next_id
    #         self.tm += 0.01
    #         return old_index
    #     else:
    #         # self.states[self.curr_capacity] = key
    #         self.hashes[self.curr_capacity] = hash
    #         # self.hashes[tuple(hash)] = self.curr_capacity
    #         # self.obses[self.curr_capacity] = obs
    #         self.q_values_decay[self.curr_capacity] = value_decay
    #         self.lru[self.curr_capacity] = self.tm
    #         if prev_id >= 0:
    #             self.next_id[prev_id] = [self.curr_capacity]
    #             self.next_action[prev_id] = [self.action_number]
    #         self.prev_id[self.curr_capacity] = [prev_id]
    #         self.prev_action[self.curr_capacity] = [prev_action]
    #         # self.next_id[self.curr_capacity] = next_id
    #         self.curr_capacity += 1
    #         self.tm += 0.01
    #         print("adding here", self.curr_capacity)
    #         return self.curr_capacity - 1


    def update_kdtree(self):
        if self.build_tree:
            del self.hash_tree
            # del self.hash_tree
        if self.curr_capacity <= 0:
            return
        print("build tree", self.curr_capacity)
        # self.tree = KDTree(self.states[:self.curr_capacity])
        self.hash_tree = KDTree(self.hashes[:self.curr_capacity])
        # self.hash_tree = KDTree(self.hashes[:self.curr_capacity])
        self.build_tree = True
        self.build_tree_times += 1
        if self.build_tree_times == 50:
            self.build_tree_times = 0
            gc.collect()
