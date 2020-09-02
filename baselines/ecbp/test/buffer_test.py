import numpy as np

import tensorflow as tf
from pyvirtualdisplay import Display
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# display = Display(visible=1, size=(1080, 768))
# display.start()
from baselines.ecbp.util import *
from baselines.ecbp.agents.graph.build_graph_contrast_target import *
from baselines.ecbp.test.value_iteration import *


def buffertest(agent,comment="None"):
    true_label = []
    label_1 = []
    label_2 = []
    label_3 = []
    representation = []
    for i in range(agent.buffer_capacity):
        label_1.append(agent.replay_buffer[i][0] + agent.replay_buffer[i][1])
        label_2.append(agent.replay_buffer[i][0] - agent.replay_buffer[i][1])
        label_3.append(agent.send_and_receive(7,i))
        true_label.append((agent.replay_buffer[i][0], agent.replay_buffer[i][1]))
        representation.append(agent.send_and_receive(6, i))

    print(label_3)
    tsne = TSNE(n_components=2, random_state=2)
    x_embeded = tsne.fit_transform(representation)
    # label_1 = np.array(label_1, dtype=np.float32)
    # label_1 = label_1 - np.min(label_1)
    # label_1 = label_1 / np.max(label_1)
    # plt.figure(figsize=(8, 8))
    # plt.scatter(np.reshape(x_embeded[:, 0], -1), x_embeded[:, 1].reshape(-1), c=label_1.reshape(-1))
    # plt.colorbar()
    # plt.savefig("./plot/{}-1.png".format(comment))
    # plt.show()

    # label_2 = np.array(label_2, dtype=np.float32)
    # label_2 = label_2 - np.min(label_2)
    # label_2 = label_2 / np.max(label_2)
    # plt.figure(figsize=(8, 8))
    # plt.scatter(np.reshape(x_embeded[:, 0], -1), x_embeded[:, 1].reshape(-1), c=label_2.reshape(-1))
    # plt.colorbar()
    # plt.savefig("./plot/{}-2.png".format(comment))
    # plt.show()

    label_3 = np.nan_to_num(np.array(label_3, dtype=np.float32))
    label_3 = label_3 - np.min(label_3)
    label_3 = label_3 / np.max(label_3)
    plt.figure(figsize=(8, 8))
    plt.scatter(np.reshape(x_embeded[:, 0], -1), x_embeded[:, 1].reshape(-1), c=label_3.reshape(-1))
    plt.colorbar()
    plt.savefig("./plot/{}-3.png".format(comment))
    plt.show()

