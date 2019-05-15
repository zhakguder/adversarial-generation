from functools import partial
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from utils import _softplus_inverse
import tensorflow_probability as tfp
from custom_layers import LSHLayer, clusterLayer
from settings import get_settings

tfd = tfp.distributions
tfpl = tfp.layers
from ipdb import set_trace

forward_calls = ''
flags, params = get_settings()

def build_net(hidden_dims):
    dense_relu = partial(Dense, activation='tanh')
    net = Sequential()
    if forward_calls in ['encoder', 'mnist']:
        prev_dim = (28, 28)
    elif forward_calls == 'decoder':
        prev_dim = params['latent_dim']

    if forward_calls in ['mnist', 'encoder']:
        net.add(Flatten(input_shape=prev_dim))
    for idx, dim in  enumerate(hidden_dims):
        net.add(dense_relu(dim, name="{}_relu_{}".format(forward_calls, idx), input_shape = [prev_dim])) #
        #print('Dim: {}'.format(prev_dim))
        prev_dim = dim
    return net

def make_encoder(hidden_dims, latent_dim, out_activation, network=None):
    global forward_calls
    forward_calls = 'encoder'
    if network is not None:
        encoder_net = network
    else:
        encoder_net = build_net(hidden_dims)
    encoder_net.add(Dense(latent_dim * 2, activation = out_activation, name = '{}_{}'.format(forward_calls, out_activation)))

    def encoder(inputs):
        outputs = encoder_net(inputs)
        return outputs
    return encoder, encoder_net

def make_decoder(hidden_dims, output_dim, network=None):
    global forward_calls
    out_activation = 'linear'
    forward_calls = 'decoder'
    if network is not None:
        decoder_net = network
    else:
        decoder_net = build_net(hidden_dims)
        decoder_net.add(Dense(output_dim, activation = out_activation, name = '{}_{}'.format(forward_calls, out_activation)))
    def decoder(sample):
        reconstruction = decoder_net(sample)
        return reconstruction
    return decoder, decoder_net


def make_lsh(dim, w):
    net = Sequential()
    net.add(LSHLayer(dim, w))
    def lsh(reconstructions):
        hash_codes = net(reconstructions)
        return hash_codes
    return lsh, net

def make_cluster():
    net = Sequential()
    net.add(clusterLayer())
    def cluster(inputs):
        q_s = net(inputs)
        return q_s
    return cluster

def make_mnist(network_dims):
    global forward_calls
    forward_calls = 'mnist'
    net = build_net(network_dims)
    net.add(Dense(10, activation='linear', trainable=False))
    return net

def initialize_eval_mnist(net):
    data = tf.random.normal((1, 784))
    net(data)
    return net

def set_mnist_weights(net, weights):
    used = 0
    for i, layer in  enumerate(net.layers):
        if i > 0:
            weight_shape = layer.weights[0].shape
            bias_shape = layer.weights[1].shape
            n_weight = tf.reduce_prod(weight_shape).numpy()
            n_bias = tf.reduce_prod(bias_shape).numpy()
            tmp_used =  used + n_weight
            layer_weights = tf.reshape(weights[used:tmp_used], weight_shape)
            used = tmp_used
            tmp_used += n_bias
            layer_biases = weights[used:tmp_used]
            used = tmp_used
            net.layers[i].set_weights([layer_weights, layer_biases])
    return net
