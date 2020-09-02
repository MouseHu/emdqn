import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch import optim
from toy.network import *


def onehot(x):
    # result = torch.zeros((len(x), num_data_point)).float()
    # for i in range(len(x)):
    #     result[i][x[i]] = 1
    # return result

    return torch.from_numpy(np.array(x)).long()


def knn_cosine(x, x0, k=4):
    assert x.shape[1] == x0.shape[1]
    x0 = x0.reshape(-1)
    dot_product = np.dot(x, x0)
    indexed_product = [(j, p) for j, p in enumerate(dot_product)]
    sorted_product = sorted(indexed_product, key=lambda x: x[1], reverse=True)
    if sorted_product[0][1] > 1 - 1e-9:
        return [sorted_product[j][0] for j in range(1, k + 1)]
    else:
        return [sorted_product[j][0] for j in range(0, k)]


def knn(x, x0, k=4):
    assert x.shape[1] == x0.shape[1]
    dist = np.sqrt(np.sum(np.square(x - x0), axis=1))
    indexed_dist = [(j, d) for j, d in enumerate(dist)]
    sorted_dist = sorted(indexed_dist, key=lambda x: x[1])
    if sorted_dist[0][1] < 1e-6:
        return [sorted_dist[j][0] for j in range(1, k + 1)]
    else:
        return [sorted_dist[j][0] for j in range(0, k)]


def threshold_knn(x, x0, k=4):
    assert x.shape[1] == x0.shape[1]
    dist = np.sqrt(np.sum(np.square(x - x0), axis=1))
    indexed_dist = [(j, d) for j, d in enumerate(dist) if d < 0.1]
    sorted_dist = sorted(indexed_dist, key=lambda x: x[1])
    if sorted_dist[0][1] < 1e-9:
        return [sorted_dist[j][0] for j in range(1, min(k + 1, len(sorted_dist)))]
    else:
        return [sorted_dist[j][0] for j in range(0, min(k, len(sorted_dist)))]


def test(buffer_x):
    error = 0
    for x, y in zip(data_x, data_y):
        x = x.reshape(1, -1)
        y_knn = knn_value_cosine(buffer_x, buffer_y, x)
        error += (y - y_knn) ** 2
    return error / num_data_point


def sample_contrastive(y, index, num_neg=20):
    target_y = y[index]
    positives = []
    negatives = []
    var = np.var(y)
    for i in range(len(index)):
        positive = np.logical_and((y < (target_y[i] + var)), (y > (target_y[i] - var))).squeeze()
        positive = np.logical_and(positive, (np.arange(len(y)) != index[i]))
        positive = np.where(positive.squeeze())[0]
        negative = np.logical_or((y > (target_y[i] + var)), (y < (target_y[i] - var)))
        negative = np.where(negative.squeeze())[0]
        # sample
        if len(positive) == 0:
            positive = [index[i]]
        positive_sample = positive[np.random.choice(len(positive), 1)]
        negative_sample = negative[np.random.choice(len(negative), num_neg)]
        positives.append(positive_sample)
        # positives.append(x[positive_sample])
        negatives.append(negative_sample)
        # negatives.append(x[negative_sample])
    # return torch.stack(positives), torch.stack(negatives)
    positives = np.array(positives).reshape(-1)
    negatives = np.array(negatives).reshape(-1)
    return onehot(positives).reshape(batch_size, 1), onehot(negatives).reshape(batch_size, num_neg)


def plot(x, y, comment):
    plt.figure(figsize=(10, 8))
    plt.scatter(np.reshape(x[:, 0], -1), x[:, 1].reshape(-1), c=y.reshape(-1))
    plt.colorbar()
    plt.savefig("./{}.png".format(comment))
    plt.show()


def regression_loss(i):
    x = onehot(np.arange(i * batch_size, (i + 1) * batch_size))
    y = torch_data_y[i * batch_size:(i + 1) * batch_size].reshape(-1)

    knn_ind = [knn_cosine(rep_buffer, rep_buffer[j].reshape(1, -1)) for j in
               range(i * batch_size, (i + 1) * batch_size)]
    # knn_ind = [knn(rep_buffer, rep_buffer[j].reshape(1, -1)) for j in range(i * batch_size, (i + 1) * batch_size)]
    knn_ind = np.array(knn_ind).reshape(-1)
    # rand_ind = np.random.choice(num_data_point, batch_size * k)
    rand_y = torch_data_y[knn_ind].reshape(batch_size, k)
    # rand_x = torch_data_x[rand_ind].reshape(batch_size, k, data_shape)
    rand_x = onehot(knn_ind)
    repr_x = repr_model(x.reshape(batch_size, 1))
    repr_rand_x = repr_model(rand_x).reshape(batch_size, k, data_shape)
    target_y = repr_model.pseudo_knn_cosine(repr_rand_x, rand_y, repr_x)

    # print(target_y.shape)
    loss = torch.nn.MSELoss()(target_y, y).sum()
    return loss


def uniformity_loss():
    x = np.random.choice(num_data_point, batch_size)
    y = np.random.choice(num_data_point, batch_size)
    repr_x = repr_model(onehot(x))
    repr_y = repr_model(onehot(y))
    g = torch.zeros(1)
    for j in range(batch_size):
        g += torch.exp(2 * torch.dot(repr_x[j], repr_y[j]) - 2)
    return g


def contrastive_loss(i):
    x = onehot(np.arange(i * batch_size, (i + 1) * batch_size))
    # x = torch_data_x[i * batch_size:(i + 1) * batch_size]
    # y = torch_data_y[i * batch_size:(i + 1) * batch_size]
    positives, negatives = sample_contrastive(data_y, np.arange(i * batch_size, (i + 1) * batch_size))
    repr_x = repr_model(x)

    repr_positive = repr_model(positives)
    repr_negative = repr_model(negatives)
    losses = []
    for b in range(batch_size):
        positive = torch.zeros(1).float()
        exp_negative = torch.zeros(1).float()
        for c in range(repr_positive.shape[1]):
            # exp_negative += np.dot(repr_x[b], repr_negative[b, c])
            positive += torch.dot(repr_x[b], repr_positive[b, c])
        for c in range(repr_negative.shape[1]):
            exp_negative += torch.exp(torch.dot(repr_x[b], repr_negative[b, c]))
        losses.append(-positive + torch.log(exp_negative))
    total_loss = torch.stack(losses).sum()
    return total_loss


def weighted_product_loss():
    u = np.random.choice(num_data_point, batch_size)
    v = np.random.choice(num_data_point, batch_size)
    repr_u = repr_model(onehot(u))
    repr_v = repr_model(onehot(v))
    u_y = torch_data_y[u]
    v_y = torch_data_y[v]
    weight = torch.abs(u_y - v_y)
    # weight = torch.min(weight, 2 - weight) - 0.5
    weight = weight ** 2
    weight = weight.squeeze()
    dot_product = [acos_safe(torch.dot(repr_u[i], repr_v[i])) for i in range(batch_size)]
    dot_product = torch.stack(dot_product)
    weighted_product = torch.dot(weight, -dot_product)
    # weighted_product = torch.nn.MSELoss()(weight*np.pi,dot_product).sum()
    return weighted_product


data_shape = 10
middle_shape = 30
num_data_point = 256
buffer_size = 256
num_epoch = 8000
batch_size = 256
k = 4
data_x = np.random.randn(num_data_point, data_shape)
data_x = data_x / np.linalg.norm(data_x, axis=1, keepdims=True)
data_y = np.random.rand(num_data_point, 1) * 2 - 1
buffer_index = np.random.choice(num_data_point, buffer_size, replace=False)
buffer_y = data_y[buffer_index]
repr_model = SimpleNet(num_data_point, data_shape, middle_shape)
# repr_model_target = SimpleNet(num_data_point,data_shape, middle_shape)

# training
optimizer = optim.SGD(repr_model.parameters(), lr=1e-4, weight_decay=0)

torch_data_x = torch.tensor(data_x, dtype=torch.float32)
torch_data_y = torch.tensor(data_y, dtype=torch.float32)
rep_buffer = np.array([repr_model(onehot([x])).detach().numpy().reshape(-1) for x in range(len(data_x))])
for epoch in range(num_epoch):
    repr_model.train()
    rep_buffer = np.array([repr_model(onehot([x])).detach().numpy().reshape(-1) for x in range(len(data_x))])
    # if epoch % 12000 == 0:
    #     plot(rep_buffer, buffer_y, "rep_buffer")
    for i in range(num_data_point // batch_size):
        optimizer.zero_grad()
        # loss1 = contrastive_loss(i)
        loss1 = weighted_product_loss()
        loss2 = uniformity_loss()
        loss = loss1 + loss2
        loss.backward()
        # print(repr_model.fc1.weight.grad)
        # print(repr_model.fc2.weight.grad)
        # print(repr_model.fc3.weight.grad)
        # print(repr_model.fc4.weight.grad)
        optimizer.step()
        print("epoch: ", epoch, " loss:", loss.item())

    # if epoch % 5 == 0:
    #     repr_model_target.load_state_dict(repr_model.state_dict())
rep_buffer = np.array([repr_model(onehot([x])).detach().numpy().reshape(-1) for x in range(len(data_x))])
# rep_buffer = np.array([repr_model(x.reshape(1, -1)).detach().numpy().reshape(-1) for x in torch_data_x])

# testing

rand_buffer = data_x[buffer_index]
rand_error = test(rand_buffer)
print("rand_error: ", rand_error)
plot(rand_buffer, buffer_y, "rand_buffer")

rep_buffer = rep_buffer[buffer_index]
rep_error = test(rep_buffer)
print("rep_error: ", rep_error)
plot(rep_buffer, buffer_y, "rep_buffer")
