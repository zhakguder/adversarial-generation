import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
from settings import get_settings

flags, params = get_settings()

if flags['app'] == 'adversarial':
  from classifier import Classifier

from ipdb import set_trace

def _scale(image, label, logits = None):
  image = tf.cast(image, tf.float32)
  image /= 255
  if logits is not None:
    return image, label, logits
  else:
    return image, label

def _reshape(image, label, logits=None):
  try:
    # if not autoencoding
    label = tf.one_hot(label, 10)
  except:
    # if autoencoding leave as is
    pass
  returns = [image, label]
  if logits is not None:
    returns.append(logits)
  return returns

def get_dataset(name='mnist', adversarial=False, adversarial_training=False):
  # Should i care about autoencoding?
  if name == 'mnist':
    from tensorflow.keras.datasets import mnist
    mnist_train_, mnist_test_ = mnist.load_data()
    mnist_train, mnist_test = mnist_train_, mnist_test_
    if adversarial:
      x_train, y_train = mnist_train_
      x_test, y_test = mnist_test_
      mnist_train = (x_train, y_train)
      if adversarial_training:
        assert not flags['classifier_train'], 'Classifier training should be done before adversarial training.'
        predictor = Classifier(flags['classifier_train'])
        logits, classes = predictor.classify(x_train)
        mnist_train = (x_train, classes, logits)
        test_logits, test_classes = predictor.classify(x_test)
        mnist_test = (x_test, test_classes, test_logits)
    batch_size = flags['data_batch_size']

    # repeat before shuffle
    train_data =tf.data.Dataset.from_tensor_slices(mnist_train).map(_scale).map(_reshape).shuffle(flags['buffer_size']).repeat().batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices(mnist_test).map(_scale).map(_reshape).shuffle(flags['buffer_size']).repeat().batch(batch_size)


  elif name == 'latent':
    dim_mix = flags['input_dim_gen']
    tfd = tfp.distributions
    quatr_mix_gauss = tfd.Mixture(
      cat=tfd.Categorical(probs=[1/2, 1/2]),
      components=[
        tfd.MultivariateNormalDiag(loc=[-10.]*dim_mix, scale_diag=[5]*dim_mix),
        tfd.MultivariateNormalDiag(loc=[10.]*dim_mix, scale_diag=[5]*dim_mix)
      ])

    train_data = tf.data.Dataset.from_tensor_slices(
    quatr_mix_gauss.sample(60000, dim_mix)).repeat().shuffle(flags['buffer_size']).batch(flags['latent_batch_size'])
    test_data = tf.data.Dataset.from_tensor_slices(
    quatr_mix_gauss.sample(10000, dim_mix)).repeat().shuffle(flags['buffer_size']).batch(flags['latent_batch_size'])

  return train_data, test_data
