import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Layer
from ipdb import set_trace
import sys
from empirical import get_cluster_qs, get_cluster_member_idx

class LSHLayer(Layer):
  def __init__(self, dim, bucket_width):
    super(LSHLayer, self).__init__()
    # https://www.mit.edu/~andoni/LSH/manual.pdf
    self.a_ = tf.random.normal([dim, 1])
    self.b_ = tf.random.uniform([1], maxval=bucket_width)
    self.bucket_width = bucket_width
    self.dim = dim

  def call(self, inputs):
      projections = tf.matmul(inputs, self.a_)
      hash_code = tf.floor((projections + self.b_)/self.bucket_width)
      hash_code = hash_code - tf.reduce_min(hash_code)
      h_code = tf.constant([int(x) for x in hash_code])
      return h_code

class clusterLayer(Layer):
    def __init__(self):
        super(clusterLayer, self).__init__()

    def call(self, inputs):
        '''
        Args:
        clusters: cluster assignments
        '''
        return get_cluster_qs(inputs)
