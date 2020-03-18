import numpy as np
from sklearn.neighbors import BallTree, KDTree
import os
from baselines.deepq.experiments.atari.lru_knn_mc import LRU_KNN_MC
import gc


class LRU_KNN_COMBINE(object):
    def __init__(self, num_actions, buffer_size, latent_dim, hash_dim, input_dims, name):
        self.ec_buffer = []
        self.num_actions = num_actions
        for i in range(num_actions):
            self.ec_buffer.append(LRU_KNN_MC(buffer_size, latent_dim, latent_dim, input_dims, 'game'))

    def add(self, action, key, hash, value_decay, prev_id=-1, prev_action=-1, obs=None):
        buffer = self.ec_buffer[action]
        if buffer.curr_capacity >= buffer.capacity:
            # find the LRU entry
            index = np.argmin(buffer.lru)
        else:
            index = buffer.curr_capacity
            buffer.curr_capacity += 1
        if buffer.prev_id[index] > 0:
            prev_buffer = self.ec_buffer[buffer.prev_action]
            prev_buffer.next_id[buffer.prev_id[index]] = -1
            prev_buffer.next_action[buffer.prev_id[index]] = -1
        buffer.states[index] = key
        buffer.hashes[index] = hash
        # self.hashes[tuple(hash)] = old_index
        buffer.obses[index] = obs
        buffer.q_values_decay[index] = value_decay
        buffer.lru[index] = buffer.tm
        if prev_id >= 0:
            prev_buffer = self.ec_buffer[prev_action]
            prev_buffer.next_id[prev_id] = index
            prev_buffer.next_action[prev_id] = action
        buffer.prev_id[index] = prev_id
        buffer.prev_action[index] = prev_action
        buffer.tm += 0.01
        return index

    def peek(self, action, z, h, value_decay, modify, verbose=False, prev_id=-1, prev_action=-1):
        buffer = self.ec_buffer[action]
        if buffer.curr_capacity == 0 or buffer.build_tree == False:
            return None, None

        dist, ind = buffer.hash_tree.query([h], k=1)
        ind = ind[0][0]
        # if self.states[ind] == key:
        # if np.allclose(self.states[ind], key):
        # if np.allclose(self.hashes[ind], h, atol=1e-08):
        if dist[0][0] < 1e-2:
            buffer.lru[ind] = buffer.tm
            buffer.tm += 0.01
            if modify:
                buffer.states[ind] = z
                if value_decay > buffer.q_values_decay[ind]:
                    buffer.q_values_decay[ind] = value_decay
                    if prev_id >= 0:
                        self.ec_buffer[buffer.prev_action[prev_id]].next_id[prev_id] = ind
                        self.ec_buffer[buffer.prev_action[prev_id]].next_action[prev_id] = action
                        buffer.prev_id[ind] = prev_id
                        buffer.prev_action[ind] = prev_action
            return buffer.q_values_decay[ind], ind, action
        # print self.states[ind], key
        else:
            pass
            # if verbose:
            #     print(dist[0])
            return None, None

    def sample(self, batch_size, K=1):
        capacity = sum([buffer.curr_capacity for buffer in self.ec_buffer])
        assert 0 < batch_size < capacity, "can't sample that much!"
        anchor_idxes = []
        anchor_actions = []
        pos_idxes = []
        pos_actions = []
        neg_idxes = []
        neg_actions = []
        while len(anchor_idxes) < batch_size:

            rand_action = np.random.randint(0, self.num_actions)
            buffer = self.ec_buffer[rand_action]
            rand_idx = np.random.randint(0, buffer.curr_capacity)
            if buffer.next_id[rand_idx] > 0:
                anchor_idxes.append(rand_idx)
                anchor_actions.append(rand_action)
                pos_idxes.append(buffer.next_id[rand_idx])
                pos_actions.append(buffer.next_action[rand_idx])
                neg_idx, neg_action = self.sample_neg_keys(
                    [(rand_idx, rand_action), (buffer.next_id[rand_idx], buffer.next_action[rand_idx])], K)
                neg_idxes.append(neg_idx)
                neg_actions.append(neg_action)
        anchor_obses = [self.ec_buffer[action].obses[id] for id, action in zip(anchor_idxes, anchor_actions)]
        anchor_keys = [self.ec_buffer[action].states[id] for id, action in zip(anchor_idxes, anchor_actions)]
        pos_keys = [self.ec_buffer[action].states[id] for id, action in zip(pos_idxes, pos_actions)]
        neg_keys = [[self.ec_buffer[action].states[id] for id, action in zip(neg_idxes[i], neg_actions[i])] for i in
                    range(len(neg_idxes))]

        anchor_places = list(zip(anchor_actions, anchor_idxes))
        pos_places = list(zip(pos_actions, pos_idxes))
        neg_places = [list(zip(neg_actions[i], neg_idxes[i])) for i in range(len(neg_idxes))]
        return anchor_places, pos_places, neg_places, \
               anchor_obses, anchor_keys, pos_keys, neg_keys

    def sample_neg_keys(self, avoids, batch_size):
        # sample negative keys
        capacity = sum([buffer.curr_capacity for buffer in self.ec_buffer])
        assert batch_size + len(
            avoids) <= capacity, "can't sample that much neg samples from episodic memory!"
        places = []
        while len(places) < batch_size:
            rand_action = np.random.randint(0, self.num_actions)
            rand_buffer = self.ec_buffer[rand_action]
            id = np.random.randint(0, rand_buffer.curr_capacity)

            if (rand_action, id) not in places:
                places.append((rand_action, id))
        return list(zip(*places))

    def update(self, places, values):
        actions, idxes = list(zip(*places))
        assert len(idxes) == len(values)
        for action, id, value in zip(actions, idxes, values):
            self.ec_buffer[action].states[id] = value

    def update_kdtree(self):
        for ec in self.ec_buffer:
            ec.update_kdtree()

    def act_value(self, action, key, h, knn, verbose=True):
        value, _ = self.peek(action, key, h, None, modify=False, verbose=verbose)
        if value is not None:
            return value, True
        else:
            # print(self.curr_capacity,knn)
            return self.ec_buffer[action].knn_value(key, knn=knn), False
