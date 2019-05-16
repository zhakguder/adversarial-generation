import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
from settings import get_settings
from data_cifar_augment import flip, color, zoom, rotate, clip
import numpy as np

_list = type([])

flags, params = get_settings()

if flags['app'] == 'adversarial':
  from classifier import Classifier

from ipdb import set_trace

def _scale(image, label, pixel_mean, logits = None,):
  image /= 255.
  if flags['dataset'] == 'cifar10':
    image -= tf.cast(pixel_mean, tf.float32) # subtract pixel mean

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
  if name in ['mnist', 'cifar10']:
    batch_size = flags['data_batch_size']
    from tensorflow.keras import datasets
    train_, test_ = datasets.mnist.load_data() if name == 'mnist' else datasets.cifar10.load_data()
    # subtract training pixel_mean from Cifar10 data
    try:
      n_train, dim1, dim2, dim3 = train_[0].shape
      reshape_shape = (1, dim1, dim2, dim3)
    except:
      n_train, dim1, dim2 = train_[0].shape
      reshape_shape = (1, dim1, dim2)
    n_test = test_[0].shape[0]
    pixel_mean =  np.mean(train_[0], axis=0).reshape(reshape_shape)/255.
    #train_, test_ = map(lambda x: tuple(_list(x) + [pixel_mean]), [train_, test_])
    x_train, x_test = train_[0].astype(np.float32), test_[0].astype(np.float32)
    train_classes, test_classes = train_[1], test_[1]

    if adversarial:
      #train, test = train_, test_

      if adversarial_training:
        assert not flags['classifier_train'], 'Classifier training should be done before adversarial training.'
        predictor = Classifier(flags['classifier_train'])

        # for adversarial training, change true class labels with classifier class predictions
        if name == 'mnist':
          (train_logits, train_classes), (test_logits, test_classes) = _list(map(lambda x:predictor.classify(x), [x_train, x_test]))
        elif name == 'cifar10':
          n_examples = 1000
          n_iter = int(n_train/n_examples)
          done = todo = 0
          train_logits,  train_classes, test_logits, test_classes = ([] for i in range(4))
          for i in range(n_iter): #change back to n_iter
            todo += n_examples
            rng = range(done, todo),
            tmp1, tmp2 = predictor.classify(x_train[rng])
            train_logits.append(tmp1)
            train_classes.append(tmp2)
            if done < n_test:
              tmp3, tmp4 = predictor.classify(x_test[rng])
              test_logits.append(tmp3)
              test_classes.append(tmp4)
            done = todo
          train_logits, train_classes, test_logits, test_classes = [np.concatenate(x) for x in [train_logits, train_classes, test_logits, test_classes]]

    train_pixel_mean = np.tile(pixel_mean, [n_train, 1,1,1])
    test_pixel_mean = np.tile(pixel_mean, [n_test, 1,1,1])
    train = (x_train, train_classes, train_pixel_mean, train_logits) if adversarial_training else [x_train, train_classes, train_pixel_mean]
    test =  (x_test, test_classes, test_pixel_mean, test_logits) if adversarial_training else (x_test, test_classes, test_pixel_mean)
    test_data = tf.data.Dataset.from_tensor_slices(test).map(_scale).map(_reshape).shuffle(flags['buffer_size']).repeat().batch(batch_size)
    train_data = tf.data.Dataset.from_tensor_slices(test).map(_scale).map(_reshape)

    if name == 'cifar10':
      # Add augmentations
      augmentations = [flip, color, zoom, rotate, clip]
      for f in augmentations:
        train_data = train_data.map(f)
    train_data = train_data.shuffle(flags['buffer_size']).repeat().batch(batch_size)

  elif name == 'latent':
    batch_size = flags['latent_batch_size']
    dim_mix = params['latent_dim']
    tfd = tfp.distributions
    quatr_mix_gauss = tfd.Mixture(
      cat=tfd.Categorical(probs=[1/2, 1/2]),
      components=[
        tfd.MultivariateNormalDiag(loc=[-10.]*dim_mix, scale_diag=[5]*dim_mix),
        tfd.MultivariateNormalDiag(loc=[10.]*dim_mix, scale_diag=[5]*dim_mix)
      ])

    train_data, test_data = map(lambda x: tf.data.Dataset.from_tensor_slices(x).shuffle(flags['buffer_size']).repeat().batch(batch_size), [quatr_mix_gauss.sample(60000, dim_mix), quatr_mix_gauss.sample(10000, dim_mix)])

  return train_data, test_data
