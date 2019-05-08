#import tensorflow.compat.v1 as tf
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
from ipdb import set_trace

def _scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label

def _reshape(image, label, autoencode=False):
  image = tf.reshape(image, [784])
  if autoencode:
    return image, image
  else:
    label = tf.one_hot(label, 10)
    return image, label

def get_dataset(name='mnist', flags=None):
  if name == 'mnist':
    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_train, mnist_test = datasets['train'], datasets['test']
    train_data = mnist_train.map(_scale).map(_reshape).repeat().shuffle(flags['buffer_size']).batch(flags['mnist_batch_size'])
    test_data = mnist_test.map(_scale).map(_reshape).repeat().shuffle(flags['buffer_size']).batch(flags['mnist_batch_size'])
  elif name == 'generator_noise':
    dim_mix = flags['input_dim_gen']
    tfd = tfp.distributions
    quatr_mix_gauss = tfd.Mixture(
      cat=tfd.Categorical(probs=[1/2, 1/2]),# 1/4, 1/4]), #1/6, 1/6, 1/8, 1/8]),
      components=[
#        tfd.MultivariateNormalDiag(loc=[-10000.]*dim_mix, scale_diag=[5000]*dim_mix),
        #tfd.MultivariateNormalDiag(loc=[-1000.]*dim_mix, scale_diag=[500]*dim_mix),
        #tfd.MultivariateNormalDiag(loc=[-100.]*dim_mix, scale_diag=[50]*dim_mix),
        tfd.MultivariateNormalDiag(loc=[-10.]*dim_mix, scale_diag=[5]*dim_mix),
        tfd.MultivariateNormalDiag(loc=[10.]*dim_mix, scale_diag=[5]*dim_mix),
        #tfd.MultivariateNormalDiag(loc=[100.]*dim_mix, scale_diag=[50]*dim_mix),
        #tfd.MultivariateNormalDiag(loc=[1000.]*dim_mix, scale_diag=[500]*dim_mix)
#        tfd.MultivariateNormalDiag(loc=[10000.]*dim_mix, scale_diag=[5000]*dim_mix)
      ])


    train_data = tf.data.Dataset.from_tensor_slices(
    quatr_mix_gauss.sample(60000, dim_mix)).repeat().shuffle(flags['buffer_size']).batch(flags['batch_size'])
    test_data = tf.data.Dataset.from_tensor_slices(
    quatr_mix_gauss.sample(10000, dim_mix)).repeat().shuffle(flags['buffer_size']).batch(flags['batch_size'])

  return train_data, test_data

def initialize_model(model, dataset, params) :
  for image in dataset.take(1):
    model = model(image, params)
  return model
