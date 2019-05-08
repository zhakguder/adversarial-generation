import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from ipdb import set_trace
from models import make_mnist, initialize_eval_mnist, set_mnist_weights
from settings import *

_, params = get_settings()

def mnist_f(weights, images, labels):
    net = make_mnist(params['mnist_network_dims'])
    initialize_eval_mnist(net)
    net = set_mnist_weights(net, weights)
    logits = net(images)
    f = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
    predicted = tf.argmax(tf.nn.softmax(logits), axis=1)
    m = tf.keras.metrics.Accuracy()
    m.update_state(tf.argmax(labels, axis=1), predicted)
    acc = m.result().numpy()
    return tf.reduce_mean(f), acc
