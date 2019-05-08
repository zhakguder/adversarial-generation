import os
import urllib
import tensorflow as tf
from losses import VAE_loss
from models import *
import sys
import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras import Model
import tensorflow_datasets as tfds
from empirical import *
from applications import mnist_f

tfd = tfp.distributions
from ipdb import set_trace

class Generator(Model):

  def __init__(self, features, params):
    super(Generator, self).__init__()
    self.output_dim = params['network_out_dim']
    self.encoder = make_encoder(params["hidden_dim"], params["latent_dim"], 'linear')
    self.decoder = make_decoder(params["hidden_dim"], self.output_dim, 1)
    self.lsh = make_lsh(self.output_dim, params["w"])
    self.cluster = make_cluster()

  def call(self, inputs):

#    latent_prior =  tfd.MultivariateNormalDiag(loc=tf.zeros([params["latent_dim"]]), scale_identity_multiplier=1.0)
#    approx_posterior = self.encoder(inputs)
#    approx_posterior_sample = tf.reshape(approx_posterior.sample(params["n_samples"]), (-1, params['latent_dim']))
    #output = self.decoder(approx_posterior_sample)
    output = self.decoder(inputs)
    hash_codes = self.lsh(output)
    proj_dict = get_clusters(output, hash_codes)
    means, distance_dict = get_cluster_means(proj_dict)
    q_s = self.cluster(distance_dict)
    return q_s, means


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    from data_2 import *
    from settings import get_settings

    flags, params = get_settings()

    # Generator network input
    train_data, test_data = get_dataset('generator_noise', flags)

    # For loss function
    mnist_train_data, mnist_test_data = get_dataset('mnist', flags)

    model = initialize_model(Generator, train_data, params)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    loss_history = []
    for batch, datae in enumerate(train_data.take(1)):
      fs = []
      accs = []
      with tf.GradientTape() as tape:
        qs, means = model(datae)
        for k in range(means.shape[0]):
          for images, labels in mnist_train_data.take(1):
            f, acc = mnist_f(means[k], images, labels)
            fs.append(f)
            accs.append(acc)
        fs = tf.stack(fs)
        loss_value = tf.math.log(qs) + fs
        loss_history.append(loss_value.numpy().mean())
        print('Loss: {}'.format(loss_history[-1]))
        print('Accuracy: {}'.format(accs))
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
