import tensorflow as tf
from contextlib import contextmanager
import time
from settings import get_settings_
from functools import wraps
from ipdb import set_trace
from time import sleep
import numpy as np

inf = np.math.inf


EPS = tf.keras.backend.epsilon()
flags, _ = get_settings_()

def _softplus_inverse(x):
  """Helper which computes the function inverse of `tf.nn.softplus`."""
  return tf.math.log(tf.math.expm1(x) + EPS)


@contextmanager
def timeit_context(name, verbose=flags['timeExecution']):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    if verbose:
      print('[{}] finished in {} m'.format(name, int(elapsedTime)/60))

def shape_tensor(tensor, tensor_w_shape):
  shape = tensor_w_shape
  tf.reshape(tensor)
  return tensor

class list_iterator:

  def __init__(self, func):
    self.func = func

  def _get_weights(self, list_):
    return [item.weights for item in list_]

  def _get_iterable(self, inputs):
    return iter(inputs)

  def _get_source(self):
    return self.source

  def _log_source(self, source):
    print('Checking {}'.format(source))

  def __call__(self, list_of_tensors, source):

    self._log_source(source)
    try:
      list_of_tensors = self._get_weights(list_of_tensors)
    except:
      pass
    iterable = self._get_iterable(list_of_tensors)
    return self.func(iterable)

@list_iterator
def check_nan(tensor_iterator):
  try:
    while not tf.math.reduce_any(tf.math.is_nan(next(tensor_iterator))):
      pass
    print ("Nan encountered!")
  except StopIteration:
      print("No NaNs in output")
  sleep(2)

@list_iterator
def min_max(tensor_iterator):
  min_ = inf
  max_ = -inf
  try:
    for item in next(tensor_iterator):
      min_ = min(tf.reduce_min(item).numpy(), min_)
      max_ = max(tf.reduce_max(item).numpy(), max_)

  except StopIteration:
    pass
  print('Min: {}, max: {}'.format(min_, max_))
  sleep(2)
