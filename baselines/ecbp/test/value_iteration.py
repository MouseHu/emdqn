import numpy as np


def value_iteration(env, gamma=0.99):
    values = np.zeros(env.observation_space.shape[0])
    transition = np.zeros((env.observation_space.shape[0], env.action_space.n))
    rewards = np.zeros((env.observation_space.shape[0], env.action_space.n))
    dones = np.zeros((env.observation_space.shape[0], env.action_space.n))
    for s in range(env.observation_space.shape[0]):
        for a in range(env.action_space.n):
            env.reset()
            env.set_state(s)
            state_tp1, reward, done, info = env.step(a)
            transition[s, a] = np.argmax(state_tp1).astype(np.int)
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

    print(values)
    return values
