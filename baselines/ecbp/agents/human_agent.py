import numpy as np
import readchar


class HumanAgent(object):
    def __init__(self, keyboard_config):
        self.keyboard_config = keyboard_config

    def act(self, obs, is_train=True):
        print("input action:",end=" ")
        char = readchar.readkey()
        # char = input("input action:")
        action = self.keyboard_config.get(char, 5)
        print(" ")
        return action

    def observe(self, action, reward, obs, done, train):
        pass
