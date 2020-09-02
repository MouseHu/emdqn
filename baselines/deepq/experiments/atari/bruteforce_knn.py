import numpy as np
import tensorflow as tf


class BruteForceKNN(object):
    def __init__(self, buffersize, dimension, X):
        self.size = buffersize
        self.dimension = dimension
        # self.X = X
        self.knn = self.build_graph()

    def build_graph(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.dimension])
        self.query = tf.placeholder(tf.float32, shape=[None, self.dimension])
        self.k = tf.placeholder(tf.int32, shape=(), name="k")
        neg_distances = tf.matmul(self.query, tf.transpose(self.X))
        # neg_distances = tf.negative(distances)
        values, indexes = tf.nn.top_k(neg_distances, self.k)

        return 1-values, indexes
