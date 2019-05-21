from functools import partial
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from utils import _softplus_inverse
import tensorflow_probability as tfp
from custom_layers import LSHLayer, clusterLayer
from settings import get_settings
from functools import reduce

tfd = tfp.distributions
tfpl = tfp.layers
from ipdb import set_trace

flags, params = get_settings()
forward_calls = ''
layer_count = 0

def build_net(hidden_dims, trainable=True):
    dense_relu = partial(Dense, activation='tanh')
    net = Sequential()
    if forward_calls in ['encoder', 'mnist']:
        prev_dim = (28, 28)
    elif forward_calls == 'decoder':
        prev_dim = params['latent_dim']

    if forward_calls in ['mnist', 'encoder']:
        net.add(Flatten(input_shape=prev_dim))
        prev_dim = reduce(lambda x,y: x*y, prev_dim)
    for idx, dim in  enumerate(hidden_dims):
        net.add(dense_relu(dim, name="{}_relu_{}".format(forward_calls, idx), input_shape = [prev_dim], trainable=trainable)) #
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
    net = build_net(network_dims, trainable=True)
    net.add(Dense(10, activation='linear', trainable=True))
    return net

def initialize_eval_network(net):
    img_dim = params['img_dim']
    dataset = flags['dataset']
    flatten = True if dataset == 'mnist' else False

    example_shape = reduce(lambda x, y: x*y, img_dim) if flatten else img_dim
    shape = [1] + list(example_shape)

    data = tf.random.normal(shape)
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


def mnist_classifier_net(input_shape, output_shape, training):
    net = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu', trainable=training),
        tf.keras.layers.Dense(256, activation='relu', trainable=training),
        tf.keras.layers.Dense(output_shape, trainable=training)
    ])
    return net

def cifar10_classifier_net(filters_array, dropout_array, input_shape, output_shape, training):
    from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout
    net = tf.keras.models.Sequential()
    layer_count = 0
    for filters, dropout in zip (filters_array, dropout_array):
        for i in range(2):
            if layer_count == 0:
                net.add(
                    Conv2D(filters, (3,3), padding='same', activation='relu', trainable=training, input_shape = input_shape))
                net.add(BatchNormalization())
                layer_count +=1
            else:
                net.add(Conv2D(filters, (3,3), padding='same', activation='relu', trainable=training))
                net.add(BatchNormalization())
                layer_count += 1
        net.add(MaxPooling2D(pool_size=(2,2)))
        net.add(Dropout(dropout))
    net.add(Flatten())
    net.add(tf.keras.layers.Dense(output_shape, trainable=training))
    return net

if __name__ == '__main__':
    net = cifar10_classifier_net([32, 64, 128], [0.2, 0.3, 0.4], (32, 32), 10, True)
