import tensorflow as tf
from contextlib import contextmanager
import time
from settings import get_settings

flags, _ = get_settings()

def _softplus_inverse(x):
  """Helper which computes the function inverse of `tf.nn.softplus`."""
  return tf.math.log(tf.math.expm1(x))


@contextmanager
def timeit_context(name, verbose=flags['timeExecution']):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    if verbose:
      print('[{}] finished in {} m'.format(name, int(elapsedTime)/60))
