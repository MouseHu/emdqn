import numpy as np


def sample_contrast(transition, batch_size, dataset):
    num_states = len(transition)
    # sample_tar = []
    sample_pos = []
    sample_neg = []
    assert num_states >= batch_size
    sample_tar = np.random.choice(np.arange(num_states), batch_size)
    for i in range(batch_size):
        sample_pos.append(np.random.choice(transition[i], 1))
        neg = np.random.randint(0, num_states)
        while sample_tar[i] in transition[neg] or neg in transition[sample_tar[i]]:
            neg = np.random.randint(0, num_states)
        sample_neg.append(neg)

    data_tar = dataset.X[np.squeeze(sample_tar)]
    data_pos = dataset.X[np.squeeze(sample_pos)]
    data_neg = dataset.X[np.squeeze(sample_neg)]

    return data_tar, data_pos, data_neg
