import numpy as np
from toy.dataset import ValueDataset


def value_iteration(env, gamma=0.99):
    try:
        num_state = env.observation_space.shape[0]
    except IndexError:
        num_state = env.observation_space.n
    values = np.zeros(num_state)
    transition = np.zeros((num_state, env.action_space.n))
    rewards = np.zeros((num_state, env.action_space.n))
    dones = np.zeros((num_state, env.action_space.n))
    for s in range(num_state):
        for a in range(env.action_space.n):
            env.reset(s)
            # env.set_state(s)
            state_tp1, reward, done, info = env.step(a)
            # print(state_tp1,s,a)
            # transition[s, a] = np.argmax(state_tp1).astype(np.int)
            transition[s, a] = state_tp1
            rewards[s, a] = reward
            dones[s, a] = done

    for _ in range(len(values)):
        for s in range(len(values)):
            q = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                q[a] = rewards[s, a]
                if not dones[s, a]:
                    q[a] += gamma * values[int(transition[s, a])]
            values[s] = np.max(q)
    # print(rewards)
    # print(transition)
    print(values)
    return values, rewards, transition


def gen_dataset_with_value_iteration(env, device):
    gamma = 0.99
    values, rewards, transition = value_iteration(env,gamma)
    q_value = np.zeros_like(transition)
    for s in range(len(q_value)):
        for a in range(len(q_value[0])):
            s_tp1 = int(transition[s,a])
            q_value[s,a] = gamma*values[s_tp1]+rewards[s,a]
    print("value variance:", np.var(values))
    obs = []
    for s in range(len(values)):
        env.reset(s)
        # env.set_state(s)
        obs.append(env.render())
    obs = np.array(obs)
    # dataset = zip(obs,values)

    return ValueDataset(obs, q_value, device), transition
