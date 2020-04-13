from env.chain_env import ChainEnv
from env.circle_env import CircleEnv
from env.mk_env import MKEnv
from env.corner_env import CornerEnv
from algo.EpisodicControl import EpisodicAgent
from algo.EpisodicControlwE import EpisodicwExplorationAgent
from algo.HumanAgent import HumanAgent

import numpy as np
import matplotlib.pyplot as plt

num_steps = 100
num_phases = 10
chain_len = 10
gamma = 0.99
# env = ChainEnv(chain_len, 0.1, 0.9, 1)
# env = CircleEnv(chain_len, 8, 0.1, 500)
env = CornerEnv(chain_len * 2, chain_len, 1,100)
# env = MKEnv(200)
agent = EpisodicwExplorationAgent(chain_len * 2 + 2, 2, 0, gamma, beta=1)
# agent = HumanAgent({"w": 0, "s": 2, "d": 1, "a": 3,"x":4,"p":5})
# agent = EpisodicAgent(chain_len * 2 + 2, 2, 0.1, gamma, beta=1)

returns = [0.0]
averages = []
start_step = 0
done = False
episode = 0


def run_one_phase(num_steps, phase):
    returns = []
    steps = 0
    steps_cummulate = []
    obses = []
    while steps < num_steps:
        rtn, step, obs = run_one_episode(train=phase)
        returns.append(rtn)
        obses.append(obs)
        steps += step
        steps_cummulate.append(step)
    return returns, obses, steps_cummulate


def run_one_episode(train=True):
    done = False
    obs = env.reset()
    step = 0
    rtn = 0
    while not done:
        action = agent.act(obs, train=train)
        if action < 0:
            print("invalid action. exiting.")
            exit(-1)

        obs, reward, done, info = env.step(action)
        agent.observe(action, reward, obs, done, train)
        rtn += (gamma ** step) * reward
        step += 1
    return rtn, step, obs


test_returns = []
train_returns = []
test_obses = []
train_obses = []
train_stepss = []
test_stepss = []
for i in range(num_phases):
    print("phase", i)
    train_return, train_obs, train_steps = run_one_phase(num_steps, True)
    test_return, test_obs, test_steps = run_one_phase(num_steps, False)
    test_returns += test_return
    train_returns += train_return
    test_obses += test_obs
    train_obses += train_obs
    train_stepss += train_steps
    test_stepss += test_steps
    # print(test_return)
for i in range(1, len(train_stepss)):
    train_stepss[i] += train_stepss[i - 1]

for i in range(1, len(test_stepss)):
    test_stepss[i] += test_stepss[i - 1]
# returns.pop()
# print(np.average(returns))
print(agent.table)
# print(agent.count)
# print(agent.intrinsic)
plt.plot(test_stepss, test_returns)
plt.show()
plt.plot(train_stepss, train_returns)
# plt.plot(np.arange(len(test_returns)), test_returns)
plt.show()
# print(np.histogram(test_obses))
# print(np.histogram(train_obses))
# plt.hist(test_obses)
# plt.show()
# plt.hist(train_obses)
# plt.show()
