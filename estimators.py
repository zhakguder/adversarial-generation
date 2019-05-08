import os
import urllib
import tensorflow.compat.v1 as tf
#import tensorflow as tf
from losses import VAE_loss
from models import *

import sys
import numpy as np
import tensorflow_probability as tfp

tfd = tfp.distributions
tf.enable_eager_execution()
from ipdb import set_trace

def VAE_fn(features, labels, mode, params, config):
  """Builds the model function for use in an estimator.
  Arguments:
    features: The input features for the estimator.
    labels: The labels, unused here.
    mode: Signifies whether it is train or test or predict.
    params: Some hyperparameters as a dictionary.
    config: The RunConfig, unused here.
  Returns:
    EstimatorSpec: A tf.estimator.EstimatorSpec instance.
  """

  del labels, config
  output_dim = features.shape[1]#.value
  encoder = make_encoder(params["hidden_dim"],
                         params["latent_dim"],
                         'linear')
  decoder = make_decoder(params["hidden_dim"],
                         output_dim,
                         1)
  lsh = make_lsh(output_dim, params["w"])

  latent_prior =  tfd.MultivariateNormalDiag(loc=tf.zeros([params["latent_dim"]]), scale_identity_multiplier=1.0)

  approx_posterior = encoder(features)
  approx_posterior_sample = approx_posterior.sample(params["n_samples"])
  decoder_likelihood = decoder(approx_posterior_sample)
  loss, elbo, avg_rate, avg_distortion = VAE_loss(features, decoder_likelihood, approx_posterior, latent_prior)

  print('we are here')
  hash_codes = lsh(decoder_likelihood.sample(1000))
  # Decode samples from the prior for visualization.
#  random_image = decoder(latent_prior.sample(16))

  # Perform variational inference by minimizing the -ELBO.
  global_step = tf.compat.v1.train.get_or_create_global_step()
  learning_rate = tf.compat.v1.train.cosine_decay(
      params["learning_rate"], global_step, params["max_steps"])
  tf.compat.v1.summary.scalar("learning_rate", learning_rate)
  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
  train_op = optimizer.minimize(loss, global_step=global_step)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops={
          "elbo":
              tf.compat.v1.metrics.mean(elbo),
          "rate":
              tf.compat.v1.metrics.mean(avg_rate),
          "distortion":
              tf.compat.v1.metrics.mean(avg_distortion),
      },
)


ROOT_PATH = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
FILE_TEMPLATE = "binarized_mnist_{split}.amat"


def download(directory, filename):
  """Downloads a file."""
  filepath = os.path.join(directory, filename)
  if tf.io.gfile.exists(filepath):
    return filepath
  if not tf.io.gfile.exists(directory):
    tf.io.gfile.makedirs(directory)
  url = os.path.join(ROOT_PATH, filename)
  print("Downloading %s to %s" % (url, filepath))
  urllib.request.urlretrieve(url, filepath)
  return filepath

def static_mnist_dataset(directory, split_name):
  """Returns binary static MNIST tf.data.Dataset."""
  amat_file = download(directory, FILE_TEMPLATE.format(split=split_name))
  dataset = tf.data.TextLineDataset(amat_file)
  str_to_arr = lambda string: np.array([c == b"1" for c in string.split()])

  def _parser(s):
    booltensor = tf.compat.v1.py_func(str_to_arr, [s], tf.bool)
    #reshaped = tf.reshape(booltensor, [28, 28, 1])
    reshaped = tf.reshape(booltensor, [784])
    return tf.cast(reshaped, dtype=tf.float32), tf.constant(0, tf.int32)

  return dataset.map(_parser)


def build_input_fns(data_dir, batch_size):
  """Builds an Iterator switching between train and heldout data."""

  # Build an iterator over training batches.
  def train_input_fn():
    dataset = static_mnist_dataset(data_dir, "train")
    dataset = dataset.shuffle(50000).repeat().batch(batch_size)
    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

  # Build an iterator over the heldout set.
  def eval_input_fn():
    eval_dataset = static_mnist_dataset(data_dir, "valid")
    eval_dataset = eval_dataset.batch(batch_size)
    return tf.compat.v1.data.make_one_shot_iterator(eval_dataset).get_next()

  return train_input_fn, eval_input_fn



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    eager = 'Executing eagerly' if tf.executing_eagerly() else 'Not eager execution'
    print(eager)

    #cuda = 'Using GPU' if tf.test.is_gpu_available() else 'Using CPU'
    #print(cuda)

    params = {}
    params['hidden_dim'] = [40,20]
    #softmax_dim = 10
    params['latent_dim'] = 10
    params['batch_size'] = 64
    params['n_samples'] = 16
    params['data_dir'] = "vae/data"
    params['learning_rate'] = 0.001
    params['max_steps'] = 200
    params['w'] = 4
    train_input_fn, eval_input_fn = build_input_fns(params['data_dir'], params['batch_size'])


    estimator = tf.estimator.Estimator(
      VAE_fn,
      params=params,
      config=tf.estimator.RunConfig(
          model_dir='vae/model',
          save_checkpoints_steps=10,
      ),
  )

    for _ in range(20):
        estimator.train(train_input_fn, steps=10)
        eval_results = estimator.evaluate(eval_input_fn)
        print("Evaluation_results:\n\t%s\n" % eval_results)
