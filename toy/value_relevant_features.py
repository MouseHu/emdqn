import numpy as np
import torch
from toy.value_iteration import *
from toy.network import AttentionNet
from toy.env.fourrooms import Fourrooms
from torch import optim
import os
import cv2
# what to do: create an env(like fourroom, compute the value of its states,
# and train a neural network with mask to see if it can learn desired mask)


lr = 1e-4
epochs = 100
batch_size = 32
log_interval = 1
device = torch.device("cuda")

env = Fourrooms()

dataset = gen_dataset_with_value_iteration(env, device)

value_network = AttentionNet(input_size=288).to(device)

optimizer = optim.Adam(value_network.parameters(), lr=lr, weight_decay=0)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True)


# train network
for epoch in range(1, epochs + 1):
    value_network.train()
    train_loss = 0
    total_correct = 0
    for batch_idx, (obs, value_gt) in enumerate(data_loader):
        optimizer.zero_grad()
        mask, value_predict = value_network(obs)
        loss = value_network.loss_func(mask, value_predict, value_gt)
        loss.sum().backward()
        # total_correct += correct.sum().item()
        train_loss += loss.sum().item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(obs), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader),
                       loss.sum().item() / len(obs), end='\r'))

    # if epoch % 1 == 0:
    #         torch.save(model.state_dict(), "./model/FC_{}_epoch{}_predict{}.model".format(b, epoch, args.predict))


# print mask
target_path = "./attention/"
# if not os.path.exists(target_path):
os.makedirs(os.path.join(target_path,"./mask/"),exist_ok=True)
os.makedirs(os.path.join(target_path,"./image/"),exist_ok=True)
os.makedirs(os.path.join(target_path,"./masked_image/"),exist_ok=True)
obs,values = dataset.X,dataset.y
for i,data in enumerate(zip(obs,values)):
    obs,value = data
    mask, value_predict = value_network(obs[np.newaxis,...])
    obs = obs.cpu().numpy().transpose((1,2,0))
    mask = mask.detach().cpu().numpy()[0,0]

    mask = cv2.resize(mask, (obs.shape[0], obs.shape[1]))
    # print(self.obs_shape)
    mask = np.repeat(mask[..., np.newaxis], 3, axis=2)
    masked_image = obs*mask
    cv2.imwrite(os.path.join(target_path,"./mask/","{}.png".format(i)),mask)
    cv2.imwrite(os.path.join(target_path,"./image/","{}.png".format(i)),obs)
    cv2.imwrite(os.path.join(target_path,"./masked_image/","{}.png".format(i)),masked_image)


