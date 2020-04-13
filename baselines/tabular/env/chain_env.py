class ChainEnv(object):
    def __init__(self, len, r0, r1, r2):
        assert len > 0
        self.len = len
        self.r0 = r0
        self.r1 = r1
        self.r2 = r2
        self.curr_state = 0

    def reset(self):
        self.curr_state = 0
        return 0

    def reward(self, state):
        if state == self.len:
            return self.r2
        elif state > self.len:
            return (state - self.len) / self.len * (self.r1 - self.r0) + self.r0
        else:
            return 0

    def step(self, action):
        assert 0 <= action <= 1
        if action == 0:
            self.curr_state += 1
            done = self.curr_state >= self.len
            reward = self.reward(self.curr_state)
            # if done:
            # print("bingo!")
            return self.curr_state, reward, done, None
        else:
            self.curr_state = self.len + 1 + self.curr_state
            reward = self.reward(self.curr_state)
            # print(self.curr_state)
            return self.curr_state, reward, True, None
