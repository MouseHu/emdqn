import numpy as np
import torch


class ValueDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, observation, value,device=torch.device("cpu")):
        super(ValueDataset, self).__init__()
        observation = observation.transpose((0,3,1,2))/255
        observation = observation - np.mean(observation)
        print("Loading data...")
        print("max of x:",np.max(observation))
        self.X = torch.tensor(observation).float().to(device)
        # self.X
        self.y = torch.tensor(np.array(value)).float().to(device)
        self.size = len(value)
        print("size:", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        assert 0 <= item < self.size
        # print(self.num_class)
        # print(self.X[item])
        return self.X[item], self.y[item]
