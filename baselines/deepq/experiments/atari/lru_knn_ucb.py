import numpy as np
from sklearn.neighbors import BallTree, KDTree
import os

import gc


# each action -> a lru_knn buffer
class LRU_KNN_UCB(object):
    def __init__(self, capacity, z_dim, env_name, mode="mean", num_actions=6):
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
        self.tree = None
        self.addnum = 0
        self.buildnum = 256
        self.buildnum_max = 256
        self.bufpath = './buffer/%s' % self.env_name
        self.build_tree_times = 0
        self.build_tree = False
        self.mode = mode

    def load(self, action):
        try:
            assert (os.path.exists(self.bufpath))
            lru = np.load(os.path.join(self.bufpath, 'lru_%d.npy' % action))
            cap = lru.shape[0]
            self.curr_capacity = cap
            self.tm = np.max(lru) + 0.01
            self.buildnum = self.buildnum_max

            self.states[:cap] = np.load(os.path.join(self.bufpath, 'states_%d.npy' % action))
            self.q_values_decay[:cap] = np.load(os.path.join(self.bufpath, 'q_values_decay_%d.npy' % action))
            self.lru[:cap] = lru
            self.tree = KDTree(self.states[:self.curr_capacity])
            print("load %d-th buffer success, cap=%d" % (action, cap))
        except:
            print("load %d-th buffer failed" % action)

    def save(self, action):
        if not os.path.exists('buffer'):
            os.makedirs('buffer')
        if not os.path.exists(self.bufpath):
            os.makedirs(self.bufpath)
        np.save(os.path.join(self.bufpath, 'states_%d' % action), self.states[:self.curr_capacity])
        np.save(os.path.join(self.bufpath, 'q_values_decay_%d' % action), self.q_values_decay[:self.curr_capacity])
        np.save(os.path.join(self.bufpath, 'lru_%d' % action), self.lru[:self.curr_capacity])

    def peek(self, key, value_decay, action=-1, modify=False):
        if self.curr_capacity == 0 or self.build_tree == False:
            return None, None, None

        dist, ind = self.tree.query([key], k=1)
        ind = ind[0][0]
        # print("peek", dist[0][0])
        # if self.states[ind] == key:
        # if np.allclose(self.states[ind], key):
        # if np.allclose(self.states[ind], key, atol=1e-08):
        if dist[0][0] < 1e-2:
            # print("peek success")
            self.lru[ind] = self.tm
            self.tm += 0.01
            if modify:
                if self.mode == "max":
                    if value_decay > self.q_values_decay[ind]:
                        self.q_values_decay[ind] = value_decay
                        if action>=0:
                            self.best_action[ind,action]=1
                elif self.mode == "mean":
                    self.q_values_decay[ind] = (value_decay + self.q_values_decay[ind] * self.count[ind]) / (
                            self.count[ind] + 1)
                self.count[ind] += 1
            return self.q_values_decay[ind], self.best_action[ind], self.count[ind]
        # print self.states[ind], key
        # if prints:
        #     print("peek", dist[0][0])
        return None, None,None

    def knn_value(self, key, knn):
        knn = min(self.curr_capacity, knn)
        if self.curr_capacity == 0 or self.build_tree == False:
            return 0.0,None, 1.0

        dist, ind = self.tree.query([key], k=knn)
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
        key = np.array(key).squeeze()
        exact_reference, action, count = self.peek(key, 0, modify=False)
        if exact_reference is not None:
            return exact_reference, action, np.sqrt(count), True
        else:
            return self.knn_value(key, knn) + (False,)

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
        # self.addnum += 1
        # if self.addnum % self.buildnum == 0:
        #    self.addnum = 0
        #    self.buildnum = min(self.buildnum * 2, self.buildnum_max)
        #    del self.tree
        #    self.tree = KDTree(self.states[:self.curr_capacity])
        #    self.build_tree_times += 1
        # if self.curr_capacity < self.buildnum:
        #    del self.tree
        #    self.tree = KDTree(self.states[:self.curr_capacity])
        #    self.build_tree_times += 1

        # if self.build_tree_times == 50:
        #    self.build_tree_times = 0
        #    gc.collect()

    def update_kdtree(self):
        # if self.curr_capacity == 0:
        #     return
        if self.build_tree:
            del self.tree
        # print(self.curr_capacity)
        self.tree = KDTree(self.states[:self.curr_capacity])
        self.build_tree = True
        self.build_tree_times += 1
        if self.build_tree_times == 50:
            self.build_tree_times = 0
            gc.collect()
