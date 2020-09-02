import numpy as np
from baselines.deepq.experiments.atari.knn_cuda_fixmem import knn
import time
from sklearn.neighbors import BallTree, KDTree
import copy
c = 32
dict_size = 10000
query_size = 4
query_max = 64
capacity = 1000
k = 4
for n in range(4):
    cur_time = time.time()
    query = np.random.rand(query_size,c).astype(np.float32)

    reference = np.random.rand(dict_size,c).astype(np.float32)
    address = knn.allocate(dict_size, c, query_max, k)
    # print(address, address.dtype)
    # address = copy.deepcopy(address)
    # print(address,address.dtype)
    # print("??????")
    print(address)
    for i in range(capacity):
        # print(i,np.array(reference[i]).shape)
        # print(i)
        knn.add(address,i,reference[i])
    print("add time:", time.time() - cur_time)
    cur_time = time.time()
    # print(address)
    # # Index is 1-based
    dist, ind = knn.knn(address,query.reshape(-1, c),
                        k, capacity)

    print(ind.shape, "time:", time.time() - cur_time)
    print(np.transpose(ind))
    print(np.transpose(dist))
    # cur_time = time.time()
    tree = KDTree(reference[:capacity])
    print("build tree time:", time.time() - cur_time)
    cur_time = time.time()
    dist, ind = tree.query(query, k=k)

    print(ind.shape, "time:", time.time() - cur_time)
    print(ind)
    print(dist)
# for n in range(4):
#     cur_time = time.time()
#     query = np.random.rand(query_size,c).astype(np.float32)
#
#     reference = np.random.rand(dict_size,c).astype(np.float32)
#     # address = knn.allocate(dict_size, c, query_size, 4)
#     # print(address, address.dtype)
#     # address = copy.deepcopy(address)
#     # print(address,address.dtype)
#     # print("??????")
#     address = n
#     print(n)
#     for i in range(dict_size):
#         # print(i,np.array(reference[i]).shape)
#         # print(i)
#         knn.add(address,i,reference[i])
#     print("add time:", time.time() - cur_time)
#     cur_time = time.time()
#     # print(address)
#     # # Index is 1-based
#     dist, ind = knn.knn(address,query.reshape(-1, c),
#                         4, dict_size)
#
#     print(ind.shape, "time:", time.time() - cur_time)
# for n in range(4):
#     cur_time = time.time()
#     query = np.random.rand(query_size, c).astype(np.float32)
#
#     reference = np.random.rand(dict_size, c).astype(np.float32)
#     tree = KDTree(reference.reshape(-1, c))
#     print("build tree time:", time.time() - cur_time)
#     cur_time = time.time()
#     dist, ind = tree.query(query.reshape(-1, c), k=4)
#
#     print(ind.shape, "time:", time.time() - cur_time)
