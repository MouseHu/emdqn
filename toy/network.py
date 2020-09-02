import torch
from torch import nn
from torch import optim
import numpy as np

def acos_safe(x, eps=1e-4):
    slope = np.arccos(1-eps) / eps
    # TODO: stop doing this allocation once sparse gradients with NaNs (like in
    # th.where) are handled differently.
    buf = torch.empty_like(x)
    good = abs(x) <= 1-eps
    bad = ~good
    sign = torch.sign(x[bad])
    buf[good] = torch.acos(x[good])
    buf[bad] = torch.acos(sign * (1 - eps)) - slope*sign*(abs(x[bad]) - 1 + eps)
    return buf

def knn_value(x, y, x0, k=4):
    assert x.shape[1] == x0.shape[1]
    assert x.shape[0] == y.shape[0]
    dist = np.sqrt(np.sum(np.square(x - x0), axis=1))
    indexed_dist = [(j, d) for j, d in enumerate(dist)]
    sorted_dist = sorted(indexed_dist, key=lambda x: x[1])
    if sorted_dist[0][1] < 1e-9:
        return y[sorted_dist[0][0]]
    else:
        result = 0
        dist = np.array([sorted_dist[i][1] for i in range(k)])
        coeff = np.exp(-dist)
        coeff = coeff / np.sum(coeff)
        for i in range(min(k, x.shape[0])):
            result += coeff[i] * y[sorted_dist[i][0]]
        return result


def knn_value_cosine(x, y, x0, k=4):
    assert x.shape[1] == x0.shape[1]
    assert x.shape[0] == y.shape[0]
    x0 = x0.reshape(-1)
    dot_product = np.dot(x, x0)
    indexed_product = [(j, p) for j, p in enumerate(dot_product)]
    sorted_product = sorted(indexed_product, key=lambda x: x[1], reverse=True)
    result = 0
    if sorted_product[0][1] > 1 - 1e-9:
        coeff = np.array([sorted_product[i][1] for i in range(1, k + 1)])
        coeff = np.exp(np.arccos(coeff))
        coeff = coeff / np.sum(coeff)
        for i in range(k):
            result += coeff[i] * y[sorted_product[i+1][0]]
    else:
        coeff = np.array([sorted_product[i][1] for i in range(k)])
        coeff = np.exp(np.emath.arccos(coeff))
        coeff = coeff / np.sum(coeff)
        for i in range(k):
            result += coeff[i] * y[sorted_product[i][0]]
    return result


class SimpleNet(nn.Module):
    def __init__(self, data_dim, latent_dim, middle_shape):
        super(SimpleNet, self).__init__()
        # self.fc1 = nn.Linear(data_dim, middle_shape)
        self.fc1 = nn.Embedding(data_dim, middle_shape)
        self.fc2 = nn.Linear(middle_shape, middle_shape)
        self.fc3 = nn.Linear(middle_shape, middle_shape)
        self.fc4 = nn.Linear(middle_shape, latent_dim)

    def forward(self, x):
        # print(x.shape)
        z = nn.ReLU()(self.fc1(x))
        # print(z.shape)
        z = nn.ReLU()(self.fc2(z))
        z = nn.ReLU()(self.fc3(z))
        # print(z.shape)
        out = self.fc4(z)
        # out = nn.Tanh()(self.fc4(z))
        # print(out.shape)
        # print(torch.norm(out, dim=1, keepdim=False).shape)
        # out = out / torch.norm(out, dim=-1, keepdim=True)
        out = torch.nn.functional.normalize(out,p=2,dim=-1)
        return out

    def pseudo_knn(self, x, y, x0):
        # repr_x = self.forward(x)
        # repr_x0 = self.forward(x0)
        dist = torch.sqrt(torch.sum((x - x0) ** 2, dim=2))
        coeff = torch.softmax(-dist, dim=1)
        # coeff = coeff / torch.sum(coeff)
        result = []
        for i in range(coeff.shape[0]):
            result.append(torch.dot(coeff[i], y[i]))
        return torch.stack(result)

    def pseudo_knn_cosine(self, x, y, x0):
        # repr_x = self.forward(x)
        # repr_x0 = self.forward(x0)
        # x0=x0.reshape(batch_size,-1)
        batch_size,k = x.shape[0],x.shape[1]
        inner_product = []
        for i in range(batch_size):
            for j in range(k):
                inner_product.append(acos_safe(torch.dot(x[i, j], x0[i,0])))
        inner_product = torch.stack(inner_product).reshape(batch_size,k)
        coeff = torch.softmax(inner_product*1e4, dim=1)
        # coeff = coeff / torch.sum(coeff)
        result = []
        for i in range(coeff.shape[0]):
            result.append(torch.dot(coeff[i], y[i]))
        return torch.stack(result)
