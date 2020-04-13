import numpy as np


class EpisodicwExplorationAgent(object):
    def __init__(self, state_size, action_size, episilon, gamma=0.99, beta=0.1):
        self.rmax = state_size * beta
        self.state_size = state_size
        self.action_size = action_size
        self.reward = np.zeros((state_size, action_size))
        self.table = np.zeros((state_size, action_size))
        self.back_pointer = [[] for _ in range(state_size)]
        self.count = np.zeros((state_size, action_size))
        self.intrinsic = self.rmax * np.ones((state_size, action_size))
        self.sequence = []
        self.obs = None
        # self.cur_count = None
        self.episilon = episilon
        self.gamma = gamma
        self.beta = beta

    def act(self, obs, train=True):
        assert 0 <= obs <= self.state_size
        self.obs = obs

        # instance_inr = np.max(self.exploration_coef(self.count[obs]))
        if train:
            q = self.intrinsic[obs, :]
        else:
            q = self.table[obs, :]
        if np.random.random() < self.episilon and train:
            return np.random.randint(0, self.action_size)
        else:
            q_max = np.max(q)
            if q_max == 0:
                return np.random.randint(0, self.action_size)
            max_action = np.where(q >= q_max - 1e-5)[0]
            action_selected = np.random.randint(0, len(max_action))
            return max_action[action_selected]

    def observe(self, action, reward, state_tp1, done, train=True):
        if not train:
            return
        self.sequence.append((self.obs, action, reward, state_tp1, done))

        if done:
            # if train:
            #     self.print_sequence()
            self.update_sequence()

    def print_sequence(self):
        action_char = [">", "^", "?"]
        for s, a, r, _, done in self.sequence:
            print(action_char[a], end=" ")
        print(" ")

    def exploration_coef(self, counts):
        return [self.rmax if count == 0 else self.beta / np.sqrt(count) for count in counts]

    def reward_update(self, s0, s, a, r_i, r_e, d, r_loop):
        old_r_e = self.table[s, a]
        old_r_i = self.intrinsic[s, a]
        r = self.reward[s, a]

        if s == s0 and d > 0:
            self.table[s, a] = max(self.table[s, a], (r + self.gamma * r_loop) / (1 - self.gamma ** d))
            d = 0
            r_loop = 0

        self.table[s, a] = max(self.table[s, a], r + self.gamma * r_e)
        self.intrinsic[s, a] = min(self.intrinsic[s, a], r_i)
        if self.intrinsic[s, a] < old_r_i or self.table[s, a] > old_r_e + 1e-7:
            r_i = max(self.intrinsic[s, :])
            if d > 0:
                r_loop = r_loop * self.gamma + r
            for stm1, atm1 in self.back_pointer[s]:
                self.reward_update(s0, stm1, atm1, r_i, self.table[s, a], d + 1, r_loop)

    def update_sequence(self):
        # rtn = 0
        # inrtn = 0
        # print("updating")
        # print(self.sequence)
        Rtn = [0]
        for s, a, r, sp, done in reversed(self.sequence):
            rtn = max(self.gamma * Rtn[-1] + r, self.gamma*np.max(self.table[sp, :])+ r)
            Rtn.append(rtn)
            # print(rtn)
            self.table[s, a] = max(self.table[s, a], Rtn[-1])
            self.reward[s, a] = r
            self.intrinsic[s, a] = min(self.intrinsic[s, a], (1 - done) * self.rmax)
            if (s, a) not in self.back_pointer[sp]:
                self.back_pointer[sp].append((s, a))
            # self.intrinsic[s, a] = (1 - self.alpha) * self.intrinsic[s, a] + self.alpha * inrtn
        Rtn.pop()
        for s, a, r, _, done in self.sequence:
            rtn = Rtn.pop()
            # print(s,a,rtn)
            self.reward_update(s, s, a, (1 - done) * self.rmax, rtn, 0, 0)
        self.sequence = []
