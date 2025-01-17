import numpy as np
from sklearn.neighbors import BallTree, KDTree
import os
import cv2
import gc


# each action -> a lru_knn buffer
class LRU_KNN_TEST(object):
    def __init__(self, capacity, z_dim, env_name, ob_dims, action = 0, mode="mean"):
        self.action = action
        self.env_name = env_name
        self.capacity = capacity
        self.keys = np.empty((capacity, z_dim), dtype=np.float32)
        self.obses = np.empty((capacity,) + ob_dims, dtype=np.uint8)
        self.q_values_decay = np.zeros(capacity)
        self.count = np.zeros(capacity)
        self.lru = np.zeros(capacity)
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

            self.keys[:cap] = np.load(os.path.join(self.bufpath, 'keys_%d.npy' % action))
            self.q_values_decay[:cap] = np.load(os.path.join(self.bufpath, 'q_values_decay_%d.npy' % action))
            self.lru[:cap] = lru
            self.tree = KDTree(self.keys[:self.curr_capacity])
            print("load %d-th buffer success, cap=%d" % (action, cap))
        except:
            print("load %d-th buffer failed" % action)

    def save(self, action):
        if not os.path.exists('buffer'):
            os.makedirs('buffer')
        if not os.path.exists(self.bufpath):
            os.makedirs(self.bufpath)
        np.save(os.path.join(self.bufpath, 'keys_%d' % action), self.keys[:self.curr_capacity])
        np.save(os.path.join(self.bufpath, 'q_values_decay_%d' % action), self.q_values_decay[:self.curr_capacity])
        np.save(os.path.join(self.bufpath, 'lru_%d' % action), self.lru[:self.curr_capacity])

    def peek(self, key, value_decay, modify, save_dir=None):
        if self.curr_capacity == 0 or self.build_tree == False:
            return None, None

        dist, ind = self.tree.query([key], k=1)
        ind = ind[0][0]

        # if self.keys[ind] == key:
        # if np.allclose(self.keys[ind], key):
        # if np.allclose(self.keys[ind], key, atol=1e-08):
        if dist[0][0] < 1e-2:
            self.lru[ind] = self.tm
            self.tm += 0.01
            if modify:
                if self.mode == "max":
                    if value_decay > self.q_values_decay[ind]:
                        self.q_values_decay[ind] = value_decay
                elif self.mode == "mean":
                    self.q_values_decay[ind] = (value_decay + self.q_values_decay[ind] * self.count[ind]) / (
                            self.count[ind] + 1)
                self.count[ind] += 1
            if save_dir:
                # print(save_dir)
                if not os.path.isdir(save_dir):
                    # print("yes")
                    os.makedirs(save_dir,exist_ok=True)

                cv2.imwrite(os.path.join(save_dir, "action{}_exact_match_q{}.png".format(self.action, self.q_values_decay[ind])),
                            self.obses[ind])
            return self.q_values_decay[ind], self.count[ind]
        # print self.keys[ind], key

        return None, None

    def knn_value(self, key, knn, save_dir=None):
        knn = min(self.curr_capacity, knn)
        if self.curr_capacity == 0 or self.build_tree == False:
            return 0.0, 1.0

        dist, ind = self.tree.query([key], k=knn)
        coeff = np.exp(dist[0])
        coeff = coeff / np.sum(coeff)
        value = 0.0
        value_decay = 0.0
        count = 0
        for j, index in enumerate(ind[0]):
            value_decay += self.q_values_decay[index] * coeff[j]
            count += self.count[index] * coeff[j]
            self.lru[index] = self.tm
            self.tm += 0.01

        q_decay = value_decay
        if save_dir:
            # print(save_dir)
            if not os.path.isdir(save_dir):
                # print("yes knn")
                os.makedirs(save_dir,exist_ok=True)
            for j, index in enumerate(ind[0]):
                cv2.imwrite(os.path.join(save_dir, "action{}_knn{}_q{}.png".format(self.action,j, self.q_values_decay[index])),
                            self.obses[index])
        return q_decay, count

    def act_value(self, key, knn,save_dir=None):
        # print(save_dir)
        exact_reference, count = self.peek(key, 0, modify=False,save_dir=save_dir)
        if exact_reference is not None:
            return exact_reference, np.sqrt(count), True
        else:
            return self.knn_value(key, knn, save_dir=save_dir) + (False,)

    def add(self, key, value_decay,obs):
        if self.curr_capacity >= self.capacity:
            # find the LRU entry
            old_index = np.argmin(self.lru)
            self.keys[old_index] = key
            self.q_values_decay[old_index] = value_decay
            self.lru[old_index] = self.tm
            self.count[old_index] = 2
            self.obses[old_index] = obs
        else:
            self.keys[self.curr_capacity] = key
            self.q_values_decay[self.curr_capacity] = value_decay
            self.lru[self.curr_capacity] = self.tm
            self.count[self.curr_capacity] = 2
            self.obses[self.curr_capacity] = obs
            self.curr_capacity += 1
        self.tm += 0.01
        # self.addnum += 1
        # if self.addnum % self.buildnum == 0:
        #    self.addnum = 0
        #    self.buildnum = min(self.buildnum * 2, self.buildnum_max)
        #    del self.tree
        #    self.tree = KDTree(self.keys[:self.curr_capacity])
        #    self.build_tree_times += 1
        # if self.curr_capacity < self.buildnum:
        #    del self.tree
        #    self.tree = KDTree(self.keys[:self.curr_capacity])
        #    self.build_tree_times += 1

        # if self.build_tree_times == 50:
        #    self.build_tree_times = 0
        #    gc.collect()

    def update_kdtree(self):
        # if self.curr_capacity == 0:
        #     return
        if self.build_tree:
            del self.tree
        print("build tree",self.curr_capacity)
        self.tree = KDTree(self.keys[:self.curr_capacity])
        self.build_tree = True
        self.build_tree_times += 1
        if self.build_tree_times == 50:
            self.build_tree_times = 0
            gc.collect()
