import numpy as np
import readchar


class HumanAgent(object):
    def __init__(self, keyboard_config):
        self.keyboard_config = keyboard_config

    def act(self, obs, is_train=True):
        print("input action:",end=" ")
        action = -1
        while action<0:
            char = readchar.readkey()
        # char = input("input action:")
            action = self.keyboard_config.get(char, -1)
            print(" ")
        return action

    def observe(self, action, reward, obs, done, train=True):
        pass
