import os
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from data_generators import combined_data_generators

from models import *
from empirical import *

tfd = tfp.distributions
from ipdb import set_trace

class Generator(Model):

  def __init__(self, params, flags):
    super(Generator, self).__init__()
    self.is_autoencoder = flags['autoencode']
    self.is_adversarial_generator = True if flags['app'] == 'adversarial' else False

    self.output_dim = params['network_out_dim']
    self.decoder = make_decoder(params["hidden_dim"], self.output_dim)
    self.lsh = make_lsh(self.output_dim, params["w"])
    self.cluster = make_cluster()
    self.loss = []
    self.params = params
    self.flags = flags

  def call(self, inputs):
    pass

  def get_input_data(self):
    for data in self.input_data.take(1):
      return data

  def _cluster_computations(self, output):
    hash_codes = self.lsh(output)
    # in projected dict enter data point idx and data point if application is adversarial else only enter data point
    value_index = 1 if self.is_adversarial_generator else 0
    proj_dict = get_clusters(output, hash_codes, value_index)
    self.means, distance_dict = get_cluster_means(proj_dict, value_index)
    self.cluster_dictionary =  proj_dict
    self.q_s = self.cluster(distance_dict)

    if self.is_adversarial_generator: # get indices of data points in each cluster
      self.cluster_idx = get_cluster_member_idx(proj_dict)

    if self.flags['verbose']:
      self._report_clusters()

  def get_cluster_qs(self):
    return self.q_s

  def _report_clusters(self ):
    print('Number of clusters: {}'.format(len(self.q_s)))

  def get_cluster_centers(self):
    return self.means

  def summary(self):
    print('Batch loss: {}'.format(self.loss[-1]))


class NetworkGenerator(Generator):
  def __init__(self, params, flags):
    super(NetworkGenerator, self).__init__(params, flags)
    assert not self.is_autoencoder, 'Use AdversarialGenerator class for autoencoding architectures'
    assert not self.is_adversarial_generator, 'Use AdversarialGenerator class for adversarial generation'
    self.input_data, self.aux_data = combined_data_generators(flags)

  # This is not used
  def get_aux_data(self):
    for data in self.aux_data.take(1):
      return data

  def call(self, inputs):
    output = self.decoder(inputs)
    self._cluster_computations(output)
    cluster_sizes = self.get_cluster_qs()
    return cluster_sizes


class AdversarialGenerator(Generator):
  def __init__(self, params, flags):
    super(AdversarialGenerator, self).__init__(params, flags)
    assert self.is_adversarial_generator, 'Check application, it\'s not adversarial.'
    assert self.is_autoencoder, 'Adversarial generation can be done with autoencoders only.'

    self.input_data = combined_data_generators(flags)
    encoder_dims = params["hidden_dim"][::-1]
    self.encoder = make_encoder(encoder_dims, params["latent_dim"], 'linear')
    self.decoder = make_decoder(params["hidden_dim"], self.output_dim)

  def get_cluster_idx(self):
    # cluster_idx is a dictionary (cluster index: [data point indices])
    return self.cluster_idx

  def call(self, inputs):
    #latent_prior =  tfd.MultivariateNormalDiag(loc=tf.zeros([self.params["latent_dim"]]), scale_identity_multiplier=1.0)
    images, labels = inputs
    approx_posterior = self.encoder(images)
    approx_posterior_sample = tf.reshape(approx_posterior.sample(self.params["latent_samples"]), (-1, self.params['latent_dim']))

    tf.concat([approx_posterior_sample, labels], axis=1)
    output = self.decoder(approx_posterior_sample)
    self._cluster_computations(output)
    cluster_sizes = self.get_cluster_qs()
    return cluster_sizes
