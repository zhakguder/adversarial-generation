import tensorflow as tf
from collections import OrderedDict
from utils import shape_tensor
from ipdb import set_trace

EPS = tf.keras.backend.epsilon()

def _softplus_inverse(x):
  """Helper which computes the function inverse of `tf.nn.softplus`."""
  return tf.math.log(tf.math.expm1(x) + EPS)

def get_clusters(inputs, clusters, value_index):
  projected_dict = {}
  for index, cluster in enumerate(zip(clusters, inputs)):
    code = cluster[0]
    output = cluster[1]
    item = [index, output] if value_index else [output] # place both index of cluster point and point itself or just the point depending on application
    projected_dict.setdefault(code.numpy(), []).append(item)
  return OrderedDict(sorted(projected_dict.items()))

def get_cluster_means(projected_dict, value_index):
    cluster_items_dict = {}
    means_dict = {}
    for k, v in projected_dict.items():
      cluster_items_dict[k] = tf.stack([x[value_index] for x in v])
      means_dict[k] = tf.reduce_mean(tf.stack([x[value_index] for x in v]), 0)
    distance_dict = {key: [] for key in means_dict.keys()}
    cluster_means = []
    for k in distance_dict.keys():
      cluster_means.append(means_dict[k])
      for i in distance_dict.keys():
        distance_dict[k].append(tf.norm(cluster_items_dict[k] - means_dict[i] + EPS, axis=1))
    return tf.stack(cluster_means), {k: tf.stack(v, 1) for k, v in distance_dict.items()}


def get_cluster_qs(distance_dict):
  dists = []
  cluster_weights = tf.nn.softmax(tf.concat([v for v in distance_dict.values()], 0))
  cluster_q = tf.reduce_mean(cluster_weights, 0)
  return cluster_q

def patch_fitness_grad(inputs, fitness_grad):
  @tf.custom_gradient
  def patch(inputs):
    def grad(dy):
      return dy + fitness_grad
    return inputs, grad
  return patch(inputs)

def get_cluster_member_idx(projected_dict):
  cluster_idx = {k: [ent[0] for ent in v] for k,v in projected_dict.items()}
  return cluster_idx
