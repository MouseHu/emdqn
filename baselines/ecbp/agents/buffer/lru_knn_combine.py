import numpy as np
from sklearn.neighbors import BallTree, KDTree
import os
from baselines.ecbp.agents.buffer.lru_knn_singleaction import LRU_KNN_SingleAction
import gc


class LRU_KNN_COMBINE(object):
    def __init__(self, num_actions, buffer_size, hash_dim, obs_shape, vector_input=False):
        self.ec_buffer = []
        self.hash_dim = hash_dim
        self.num_actions = num_actions

        for i in range(num_actions):
            self.ec_buffer.append(LRU_KNN_SingleAction(buffer_size, hash_dim, 'game'))

        self.replay_buffer = np.empty((buffer_size, num_actions) + obs_shape,
                                      np.float32 if vector_input else np.uint8)

    def capacity(self):
        return [buffer.curr_capacity for buffer in self.ec_buffer]

    def add(self, action, hash, value_decay, prev_id=-1, prev_action=-1):
        buffer = self.ec_buffer[action]
        if buffer.curr_capacity >= buffer.capacity:
            # find the LRU entry
            index = int(np.argmin(buffer.lru))
        else:
            index = buffer.curr_capacity
            buffer.curr_capacity += 1
        for old_prev_id, old_prev_action in zip(buffer.prev_id[index], buffer.prev_action[index]):
            prev_buffer = self.ec_buffer[old_prev_action]
            next_places = zip(prev_buffer.next_id[old_prev_id], prev_buffer.next_action[old_prev_id])
            next_places = [x for x in next_places if x != (index, action)]
            prev_buffer.next_id[old_prev_id], prev_buffer.next_action[old_prev_id] = zip(*next_places)
        buffer.hashes[index] = hash
        # buffer.obses[index] = obs
        buffer.q_values_decay[index] = value_decay
        buffer.lru[index] = buffer.tm
        if prev_id >= 0:
            assert prev_action >= 0, "id and action must be provided together"
            prev_buffer = self.ec_buffer[prev_action]
            prev_buffer.next_id[prev_id].append(index)
            prev_buffer.next_action[prev_id].append(action)
        buffer.prev_id[index] = [prev_id]
        buffer.prev_action[index] = [prev_action]
        buffer.tm += 0.01
        return index

    def peek(self, action, h, value_decay, modify, prev_id=-1, prev_action=-1):
        h = np.squeeze(h)
        buffer = self.ec_buffer[action]
        if buffer.curr_capacity == 0 or buffer.build_tree == False:
            return None, None
        dist, ind = buffer.hash_tree.query([h], k=1)
        ind = ind[0][0]
        if dist[0][0] < 1e-9:
            buffer.lru[ind] = buffer.tm
            buffer.tm += 0.01
            if modify:
                # buffer.states[ind] = z
                if value_decay > buffer.q_values_decay[ind]:
                    buffer.q_values_decay[ind] = value_decay
                    if prev_id >= 0:
                        assert prev_action >= 0, "id and action must be provided together"
                        prev_buffer = self.ec_buffer[prev_action]
                        if (action, ind) not in zip(prev_buffer.next_id[prev_id], prev_buffer.next_action[prev_id]):
                            self.ec_buffer[prev_action].next_id[prev_id].append(ind)
                            self.ec_buffer[prev_action].next_action[prev_id].append(action)
                        if (prev_action, prev_id) not in zip(buffer.prev_id[ind], buffer.prev_action[ind]):
                            buffer.prev_id[ind].append(prev_id)
                            buffer.prev_action[ind].append(prev_action)
            return buffer.q_values_decay[ind], ind
        # print self.states[ind], key
        else:
            pass
            # if verbose:
            #     print(dist[0])
            return None, None

    def sample(self, batch_size, num_neg=1):
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
            if len(buffer.next_id[rand_idx]) > 0:
                anchor_idxes.append(rand_idx)
                anchor_actions.append(rand_action)
                rand_pos = np.random.randint(0, len(buffer.next_id[rand_idx]))
                pos_idxes.append(buffer.next_id[rand_idx][rand_pos])
                pos_actions.append(buffer.next_action[rand_idx][rand_pos])
                prev_place = list(zip(buffer.prev_id[rand_idx], buffer.prev_action[rand_idx]))
                next_place = list(zip(buffer.next_id[rand_idx], buffer.next_action[rand_idx]))
                neg_action, neg_idx = self.sample_neg_keys(
                    [(rand_idx, rand_action)] + prev_place + next_place, num_neg)
                neg_idxes.append(neg_idx)
                neg_actions.append(neg_action)
        neg_idxes = np.array(neg_idxes).reshape(-1)
        neg_actions = np.array(neg_actions).reshape(-1)
        # anchor_obses = [self.ec_buffer[action].obses[id] for id, action in zip(anchor_idxes, anchor_actions)]
        # anchor_keys = [self.ec_buffer[action].hashes[id] for id, action in zip(anchor_idxes, anchor_actions)]
        # pos_keys = [self.ec_buffer[action].hashes[id] for id, action in zip(pos_idxes, pos_actions)]
        # neg_keys = [[self.ec_buffer[action].hashes[id] for id, action in zip(neg_idxes[i], neg_actions[i])] for i in
        #             range(len(neg_idxes))]

        anchor_obs = [self.replay_buffer[s, a] for a, s in zip(anchor_actions, anchor_idxes)]
        neg_obs = [self.replay_buffer[s, a] for a, s in zip(neg_actions, neg_idxes)]
        pos_obs = [self.replay_buffer[s, a] for a, s in zip(pos_actions, pos_idxes)]
        anchor_values = [self.ec_buffer[action].q_values_decay[index] for action, index in
                         zip(anchor_actions, anchor_idxes)]
        return anchor_obs, pos_obs, neg_obs, anchor_values, anchor_actions

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

    def update(self, actions, idxes, values):
        assert len(idxes) == len(values)
        for action, id, value in zip(actions, idxes, values):
            self.ec_buffer[action].hashes[id] = value

    def update_kdtree(self):
        for ec in self.ec_buffer:
            ec.update_kdtree()

    def knn_value(self, action, key, knn):
        return self.ec_buffer[action].knn_value(key, knn=knn)

    def act_value(self, action, key, knn):

        value, _ = self.peek(action, key, None, modify=False)
        if value is not None:
            return value, True
        else:
            # print(self.curr_capacity,knn)
            return self.knn_value(action, key, knn=knn), False

    def update_sequence(self, sequence, gamma):
        prev_id, prev_action = -1, -1
        Rtd = 0
        id_sequence = []
        action_sequence = []
        for obs, z, a, r, done in reversed(sequence):
            # print(np.mean(z))
            Rtd = gamma * Rtd + r
            qd, current_id = self.peek(a, z, Rtd, True, prev_id, prev_action)
            if qd is None:  # new action
                current_id = self.add(a, z, Rtd, prev_id, prev_action)

            # print(self.ec_buffer[a].capacity)
            self.replay_buffer[current_id, a] = obs
            # if prev_action >= 0 and prev_id >= 0:
            #     self.ec_buffer[prev_action].add_next(prev_id, current_id, a)
            prev_id = current_id
            prev_action = a
            id_sequence.append(current_id)
            action_sequence.append(a)
        # self.sequence = []
        # print(id_sequence)
        # print(action_sequence)
        return
