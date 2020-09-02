import numpy as np
import readchar


class HumanAgent(object):
    def __init__(self, keyboard_config):
        self.keyboard_config = keyboard_config

    def act(self, obs, train=True):

        print("input action:",end=" ")
        char = readchar.readchar()
        action = self.keyboard_config.get(char, -1)
        return action

    def observe(self, action, reward, obs, done, traina):

        pass
