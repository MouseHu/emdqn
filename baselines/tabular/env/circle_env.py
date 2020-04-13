class CircleEnv(object):
    def __init__(self, len, r0, r1, max_step):
        assert len > 0
        self.len = len
        self.r0 = r0
        self.r1 = r1
        self.curr_state = 0
        self.max_step = max_step
        self.step_count = 0

    def reset(self):
        self.curr_state = 0
        self.step_count = 0
        return 0

    def reward(self, state):
        if state < self.len:
            return self.r1
        elif state >= self.len:
            return self.r0
        else:
            return 0

    def step(self, action):
        assert 0 <= action <= 1
        self.step_count += 1
        if action == 0:
            self.curr_state += 1
            self.curr_state = self.curr_state % self.len
            done = self.step_count >= self.max_step
            reward = self.reward(self.curr_state)
            # if done:
            # print("bingo!")
            return self.curr_state, reward, done, None
        else:
            self.curr_state = self.len + 1 + self.curr_state
            reward = self.reward(self.curr_state)
            # print(self.curr_state)
            return self.curr_state, reward, True, None
