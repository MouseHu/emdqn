import torch
from torch import nn
from torch import optim
import numpy as np


def acos_safe(x, eps=1e-4):
    slope = np.arccos(1 - eps) / eps
    # TODO: stop doing this allocation once sparse gradients with NaNs (like in
    # th.where) are handled differently.
    buf = torch.empty_like(x)
    good = abs(x) <= 1 - eps
    bad = ~good
    sign = torch.sign(x[bad])
    buf[good] = torch.acos(x[good])
    buf[bad] = torch.acos(sign * (1 - eps)) - slope * sign * (abs(x[bad]) - 1 + eps)
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
            result += coeff[i] * y[sorted_product[i + 1][0]]
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
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
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
        batch_size, k = x.shape[0], x.shape[1]
        inner_product = []
        for i in range(batch_size):
            for j in range(k):
                inner_product.append(acos_safe(torch.dot(x[i, j], x0[i, 0])))
        inner_product = torch.stack(inner_product).reshape(batch_size, k)
        coeff = torch.softmax(inner_product * 1e4, dim=1)
        # coeff = coeff / torch.sum(coeff)
        result = []
        for i in range(coeff.shape[0]):
            result.append(torch.dot(coeff[i], y[i]))
        return torch.stack(result)


class AttentionNet(nn.Module):
    def __init__(self, input_size, device):
        super(AttentionNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.padding1 = nn.ReflectionPad2d(2)
        self.padding2 = nn.ReflectionPad2d(1)
        self.padding3 = nn.ReflectionPad2d(1)
        self.attention_conv = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1)
        self.normalize_attention = True
        self.coordinate = torch.arange(-input_size // 2, input_size // 2, requires_grad=False) / (input_size + 0.0)
        print(self.coordinate)
        self.mesh_x, self.mesh_y = torch.meshgrid([self.coordinate, self.coordinate])
        self.mesh_x = torch.unsqueeze(torch.unsqueeze(self.mesh_x, 0), 0).to(device)
        # self.mesh_x
        self.mesh_y = torch.unsqueeze(torch.unsqueeze(self.mesh_y, 0), 0).to(device)
        # self.value_fc_1 = nn.Linear(input_size, 32)
        # self.value_fc_2 = nn.Linear(32, 64)
        # self.value_fc_3 = nn.Linear(64, 128)
        self.value_fc_4 = nn.Linear(input_size ** 2, 4)
        self.contra_fc = nn.Linear(input_size ** 2, 32)
        self.input_size = input_size
        self.device = device
        self.zero = torch.tensor([0.], requires_grad=False).to(self.device)

    def forward(self, x):
        batch_size = x.shape[0]
        feature_map_1 = nn.ReLU()(self.padding1(self.conv1(x)))
        feature_map_2 = nn.ReLU()(self.padding2(self.conv2(feature_map_1)))
        feature_map_3 = nn.ReLU()(self.padding3(self.conv3(feature_map_2)))

        feature_max, _ = torch.max(feature_map_3, dim=1, keepdim=True)
        feature_mean = torch.mean(feature_map_3, dim=1, keepdim=True)
        feature_x = self.mesh_x.repeat(batch_size, 1, 1, 1)
        feature_y = self.mesh_y.repeat(batch_size, 1, 1, 1)
        attention_feature = torch.cat([feature_max, feature_mean, feature_x, feature_y], dim=1)
        attention_mask = nn.Sigmoid()(self.attention_conv(attention_feature))

        # if self.normalize_attention:
        #     attention_max = attention_mask.max()
        #     attention_min = attention_mask.min()
        #     attention_mask = (attention_mask - attention_min) / (attention_max - attention_min + 1e-12)
        out_feature = attention_mask * feature_max
        out_feature_flatten = torch.flatten(out_feature, start_dim=1)
        # print(out_feature_flatten.shape)
        # value_latent = nn.ReLU()(self.value_fc_1(out_feature_flatten))
        # value_latent = nn.ReLU()(self.value_fc_2(value_latent))
        # value_latent = nn.ReLU()(self.value_fc_3(value_latent))
        value = self.value_fc_4(out_feature_flatten)
        return attention_mask, value, out_feature

    def contrast_loss_func(self, obs_tar, obs_pos, obs_neg):
        # print(obs_tar.shape)
        feature_tar, feature_pos, feature_neg = self(obs_tar)[2], self(obs_pos)[2], self(obs_neg)[2]
        # print(feature_tar)
        feature_tar = torch.flatten(feature_tar, start_dim=1)
        feature_pos = torch.flatten(feature_pos, start_dim=1)
        feature_neg = torch.flatten(feature_neg, start_dim=1)
        feature_tar, feature_pos, feature_neg = self.contra_fc(feature_tar), self.contra_fc(
            feature_pos), self.contra_fc(feature_neg)

        # print(feature_tar.shape)

        def l2_dist(x, y):
            return torch.sqrt(torch.mean((x - y) ** 2, dim=1) + 1e-12)

        dist_diff = torch.max(self.zero, l2_dist(feature_tar, feature_neg) - l2_dist(feature_tar, feature_pos) + 1.)
        dist_diff_sym = torch.max(self.zero, l2_dist(feature_pos, feature_neg) - l2_dist(feature_tar, feature_pos) + 1.)
        # contrast_loss = torch.max(torch.tensor([0.],requires_grad=False).to(self.device),dist_diff) + torch.max(torch.tensor([0.],requires_grad=False).to(self.device),dist_diff_sym)
        contrast_loss = dist_diff + dist_diff_sym
        # print(contrast_loss.shape)
        # contrast_loss = feature_tar.sum() + feature_pos.sum() + feature_neg.sum()
        return contrast_loss.mean()

    def loss_func(self, attention, value_p, value_gt):
        value_p = value_p.squeeze()
        # encoder_loss = torch.mean((value_p - value_gt) ** 2)
        # encoder_loss = nn.MSELoss()(value_p,value_gt) +
        decoder_loss = nn.MSELoss()(value_p, value_gt)
        if self.normalize_attention:
            attention_max = attention.max()
            attention_min = attention.min()
            attention = (attention - attention_min) / (attention_max - attention_min + 1e-12)
        attention = torch.flatten(attention, start_dim=1)
        # attention_var= torch.var(attention)
        encoder_loss = torch.norm(attention, p=1,dim=1) / self.input_size**2

        # total_loss = encoder_loss + 1e-2 * decoder_loss

        return encoder_loss.mean(), decoder_loss
