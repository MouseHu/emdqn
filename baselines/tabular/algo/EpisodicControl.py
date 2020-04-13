import numpy as np


class EpisodicAgent(object):
    def __init__(self, state_size, action_size, episilon, gamma=0.99, alpha=0.1, beta=0.1):
        self.rmax = state_size * beta
        self.state_size = state_size
        self.action_size = action_size
        self.table = np.zeros((state_size, action_size))
        self.count = np.zeros((state_size, action_size))
        self.intrinsic = self.rmax * np.ones((state_size, action_size))
        self.sequence = []
        self.obs = None
        # self.cur_count = None
        self.episilon = episilon
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def act(self, obs, train=True):
        assert 0 <= obs <= self.state_size
        self.obs = obs

        instance_inr = np.max(self.exploration_coef(self.count[obs]))
        q = self.table[obs] + self.intrinsic[obs] + instance_inr
        if np.random.random() < self.episilon and train:
            return np.random.randint(0, self.action_size)
        else:
            q_max = np.max(q)
            max_action = np.where(q >= q_max - 1e-5)[0]
            action_selected = np.random.randint(0, len(max_action))
            return max_action[action_selected]

    def observe(self, action, reward, state_tp1, done, train=True):
        if not train:
            return
        self.sequence.append((self.obs, action, reward, state_tp1))

        if done:
            self.update_sequence()

    def exploration_coef(self, counts):
        return [self.rmax if count == 0 else self.beta / np.sqrt(count) for count in counts]

    def update_sequence(self):
        rtn = 0
        inrtn = 0
        # print("updating")
        for s, a, r, _ in reversed(self.sequence):
            rtn = self.gamma * rtn + r
            inrtn = self.gamma * inrtn + np.max(self.exploration_coef(self.count[s]))
            # print("s,a,r,intrinsic ",s,a,r,self.beta/np.sqrt(self.count[s, a]),self.intrinsic[s, a])
            self.count[s, a] += 1
            # print(s,a,r)
            self.table[s, a] = max(self.table[s, a], rtn)
            self.intrinsic[s, a] = (1 - self.alpha) * self.intrinsic[s, a] + self.alpha * inrtn
        self.sequence = []
