from ipdb import set_trace
import tensorflow as tf
def _softplus_inverse(x):
  """Helper which computes the function inverse of `tf.nn.softplus`."""
  return tf.math.log(tf.math.expm1(x))

def get_clusters(inputs, clusters):
  projected_dict = {}
  for index, cluster in enumerate(zip(clusters, inputs)):
    code = cluster[0]
    output = cluster[1]
    projected_dict.setdefault(code.numpy(), []).append([index, output])
  return projected_dict

def get_cluster_means(projected_dict):
    cluster_assignments_dict = {}
    means_dict = {}
    for k, v in projected_dict.items():
      cluster_assignments_dict[k] = tf.stack([x[1] for x in v])
      means_dict[k] = tf.reduce_mean(tf.stack([x[1] for x in v]), 0)
    distance_dict = {key: [] for key in means_dict.keys()}
    cluster_means = []
    for k in distance_dict.keys():
      cluster_means.append(means_dict[k])
      for i in distance_dict.keys():
        distance_dict[k].append(tf.norm(cluster_assignments_dict[k] - means_dict[i], axis=1))
    return tf.stack(cluster_means), distance_dict

def get_cluster_qs(distance_dict):
  distance_all_dict = {k: tf.stack(v, 1) for k, v in distance_dict.items()}
  dists = []
  cluster_weights = tf.nn.softmax(tf.concat([v for v in distance_all_dict.values()], 0))
  cluster_q = tf.reduce_mean(cluster_weights, 0)
  return cluster_q
