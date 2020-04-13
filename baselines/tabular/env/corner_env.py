class CornerEnv(object):
    def __init__(self, len, corner, r1, max_step):
        assert len > 0
        assert corner < len
        self.len = len
        self.corner = corner
        self.r1 = r1
        self.curr_state = 0
        self.max_step = max_step
        self.step_count = 0

    def reset(self):
        self.curr_state = 0
        self.step_count = 0
        return 0

    def reward(self, state):
        if state == self.len - 1:
            return self.r1
        else:
            return 0

    def step(self, action):
        assert 0 <= action <= 1
        self.step_count += 1
        if action == 0:
            if self.curr_state < self.corner:
                self.curr_state += 1
            return self.curr_state, self.reward(
                self.curr_state), False if self.step_count < self.max_step else True, None
        else:
            if self.curr_state >= self.corner:
                self.curr_state += 1
            return self.curr_state, self.reward(
                self.curr_state), self.curr_state == self.len - 1 if self.step_count < self.max_step else True, None
