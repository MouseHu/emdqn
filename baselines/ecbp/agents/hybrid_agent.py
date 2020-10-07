import numpy as np
import readchar


class HybridAgent(object):
    def __init__(self, agent1, agent2, episodes=20):
        # self.keyboard_config = keyboard_config
        self.agent1 = agent1
        self.agent2 = agent2
        self.episodes = episodes
        self.episode_count = 0

    def act(self, obs, is_train=True):
        # print("input action:",end=" ")
        if is_train and self.episode_count < self.episodes:
            self.agent1.act(obs, is_train)
            return self.agent2.act(obs, is_train)
        else:
            # self.agent1.act(obs, is_train)
            return self.agent1.act(obs, is_train)
        # return action

    def observe(self, action, reward, obs, done, train):
        if done:
            self.episode_count += 1
        self.agent1.observe(action, reward, obs, done, train)
        self.agent2.observe(action, reward, obs, done, train)


    def finish(self):
        pass

class HybridAgent2(object):
    '''
    agent1 : ai agent
    agent2 : human agent
    '''

    def __init__(self, agent1, agent2, episodes=20):
        # self.keyboard_config = keyboard_config
        self.agent1 = agent1
        self.agent2 = agent2
        self.episodes = episodes
        self.episode_count = 0

    def act(self, obs, is_train=True):
        # print("input action:",end=" ")
        if not is_train and self.episode_count == 0:

            self.agent2.act(obs, is_train,debug=True)
            return self.agent1.act(obs, is_train)
        else:
            # self.agent1.act(obs, is_train)
            return self.agent2.act(obs, is_train)

        # return action

    def observe(self, action, reward, obs, done, train):

        self.agent1.observe(action, reward, obs, done, train)
        self.agent2.observe(action, reward, obs, done, train)

        if done:
            if not train:
                self.episode_count -= 1
                if self.episode_count <= 0:
                    try:
                        self.episode_count = int(input("num episode to auto run: "))
                    except:
                        self.episode_count = 0


    def finish(self):
        pass

