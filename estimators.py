import os
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model, models
from data_generators import combined_data_generators
from utils import _softplus_inverse, check_nan, min_max, list_iterator

import numpy as np
from models import *
from empirical import *
from settings import get_settings

tfd = tfp.distributions
from ipdb import set_trace

flags, params = get_settings()

class Generator(Model):

  def __init__(self):
    super(Generator, self).__init__()
    self.is_autoencoder = flags['autoencode']
    self.is_adversarial_generator = True if flags['app'] == 'adversarial' else False
    self.output_dim = params['network_out_dim']
    self.decoder, self.decoder_net = make_decoder(params["hidden_dim"], self.output_dim)
    self.lsh, self.lsh_layer = make_lsh(self.output_dim, params["w"])
    if flags['load_checkpoint']:
      self.lsh_layer.layers[0].a_, self.lsh_layer.layers[0].b_ = tf.constant(np.load(flags['checkpoint_path']+'_a.npy')), tf.constant(np.load(flags['checkpoint_path']+'_b.npy'))
    else:
      np.save(flags['checkpoint_path']+'_a', self.lsh_layer.layers[0].a_.numpy(), allow_pickle=False)
      np.save(flags['checkpoint_path']+'_b' ,self.lsh_layer.layers[0].b_.numpy(), allow_pickle=False)
    self.cluster = make_cluster()
    self.loss = []
    self.params = params
    self.flags = flags
    self.model_path =  flags['generator_path']
    print('Generated output size: {}'.format(self.output_dim))

  def call(self, inputs):
    pass


  def cluster_quality(self, output):

    cluster_sizes = self.get_cluster_qs()
    return cluster_sizes

  def get_input_data(self):
    return self.input_data

  def cluster_computations(self, output):
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
    return self.q_s

  def get_cluster_qs(self):
    return self.q_s

  def _report_clusters(self ):
    print('Number of clusters: {}'.format(len(self.q_s)))

  def get_cluster_centers(self):
    return self.means

  def summary(self):
    print('Batch loss: {}'.format(self.loss[-1]))

  def save(self):

    for k, v in self.network_parts.items():
      models.save_model(v, flags['app'] + '_' + self.model_path + '_' + k + '.m')
    print('Saved: {}'.format(v.layers[2].weights[-1][-1]))
  def load(self):
    for k in self.network_parts.keys():
      self.network_parts[k] = models.load_model(flags['app'] + '_' + self.model_path + '_' + k + '.m')
    self.decoder, self.decoder_net = make_decoder(params["hidden_dim"], self.output_dim, network=self.network_parts['decoder'])
    try:
      self.encoder, self.encoder_net = make_encoder(encoder_dims, params["latent_dim"], 'linear', network=self.network_parts['encoder'])
    except:
      pass
    print('Loaded: {}'.format(self.decoder_net.layers[2].weights[-1][-1]))

class NetworkGenerator(Generator):
  def __init__(self):
    super(NetworkGenerator, self).__init__()
    assert not self.is_autoencoder, 'Use AdversarialGenerator class for autoencoding architectures'
    assert not self.is_adversarial_generator, 'Use AdversarialGenerator class for adversarial generation'
    self.input_data, self.aux_data = combined_data_generators(flags)
    self.network_parts = {'decoder': self.decoder_net}
  # This is not used
  def get_aux_data(self):
    return self.aux_data

  def call(self, inputs):
    # TODO: set predictor weights
    output = self.network_parts['decoder'](inputs)

    self.output_ = output
    self.cluster_computations(output)
    return output

  def patch_fitness_grad(self, fitness_grads):
    outputs = patch_fitness_grad([self.output_, fitness_grads])
    return outputs

class AdversarialGenerator(Generator):
  def __init__(self):
    super(AdversarialGenerator, self).__init__()
    assert self.is_adversarial_generator, 'Check application, it\'s not adversarial.'
    assert self.is_autoencoder, 'Adversarial generation can be done with autoencoders only.'
    self.latent_dim = params['latent_dim']
    self.input_data = combined_data_generators(flags)
    encoder_dims = params["hidden_dim"][::-1]
    self.encoder, self.encoder_net = make_encoder(encoder_dims, params["latent_dim"], 'linear')
    self.training_adversarial = flags['train_adversarial']
    self.n_classes = params['classifier_n_classes']
    self.latent_prior = tfd.MultivariateNormalDiag([0.] * self.latent_dim, scale_identity_multiplier=[1.] * flags['data_batch_size'])
    self.network_parts = {'encoder': self.encoder_net, 'decoder': self.decoder_net}
    self.approx_posterior_layer = tfpl.DistributionLambda(make_distribution_fn=lambda ts: tfd.MultivariateNormalDiag(loc=ts[..., :self.latent_dim], scale_diag=tf.nn.softplus(ts[..., self.latent_dim:] + _softplus_inverse(1.0)), name="code"))

  def get_cluster_idx(self):
    # cluster_idx is a dictionary (cluster index: [data point indices])
    return self.cluster_idx


  def call(self, inputs):
    #latent_prior =  tfd.MultivariateNormalDiag(loc=tf.zeros([self.params["latent_dim"]]), scale_identity_multiplier=1.0)
    images, labels = inputs
    output = self.network_parts['encoder'](images)
    self.approx_posterior = self.approx_posterior_layer(output)
    approx_posterior_sample = tf.reshape(self.approx_posterior.sample(self.params["latent_samples"]), (-1, self.latent_dim))

    combined_decoder_input = tf.concat([approx_posterior_sample, labels], axis=1)
    output = self.network_parts['decoder'](combined_decoder_input)
    self.cluster_computations(output)
    cluster_sizes = self.get_cluster_qs()
    return cluster_sizes

  def get_latent_dists(self):
    return self.latent_prior, self.approx_posterior
