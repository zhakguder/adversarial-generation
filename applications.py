import tensorflow as tf


from tensorflow.keras.layers import Layer, Dense
import tensorflow_probability as tfp
from models import make_mnist, initialize_eval_mnist, set_mnist_weights
from settings import *
from classifier import Classifier
import numpy as np

from ipdb import set_trace
try:
    import matplotlib.pyplot as plt
except:
    pass

tfd = tfp.distributions
flags, params = get_settings()

EPS = tf.keras.backend.epsilon()

class Eval:
    def __init__(self):
        self.metrics = {'f': [], 'accuracy': []}

    def _eval_single_cluster(self):
        pass
    def eval_clusters(self):
        pass
    def _get_data(self):
        pass
    def get_cluster_metrics(self):
        return self.metrics
    def _tidy_metrics(self):
        self.metrics = {k: tf.stack(v) for k,v in self.metrics.items()}
    def clean_eval_memory(self):
        self.metrics = {k: [] for k,v in self.metrics.items()}
    def summary(self):
        for k, v in self.get_cluster_metrics().items():
            if len(v) > 0:
                try:
                    print('{}: {}'.format(k,tf.stack(v).numpy()))
                except:
                    print('{}: {}'.format(k,v))


class NetworkEval(Eval):
    def __init__(self, data_generator):
        super(NetworkEval, self).__init__()
        self._data = data_generator
        self._accuracy = tf.keras.metrics.Accuracy()
        self.accuracy = []

    def _eval_single_cluster(self, weights, images, labels):
        net = set_mnist_weights(self.net, weights)
        logits = net(images)
        f = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        predicted = tf.argmax(tf.nn.softmax(logits), axis=1)
        self._accuracy.update_state(tf.argmax(labels, axis=1), predicted)
        self.metrics['accuracy'].append( self._accuracy.result())
        return tf.reduce_mean(f)

    def _get_data(self):
        for images, labels in self._data.take(1):
            return images, labels

    def eval_clusters(self, cluster_means):
        self.clean_eval_memory()
        n_clusters = cluster_means.shape[0]
        if flags['verbose']:
            print('Output mnist network dim: {}'.format(cluster_means.shape[1]))
        for i in range(n_clusters):
            images, labels = self._get_data()
            self.metrics['f'].append(self._eval_single_cluster(cluster_means[i], images, labels))
        self._tidy_metrics()


class MnistEval(NetworkEval):
    def __init__(self, data_generator):
        super(MnistEval, self).__init__()
        net = make_mnist(params['mnist_network_dims'])
        self.net = initialize_eval_mnist(net)

class Cifar10Eval(NetworkEval):
    def __init__(self, data_generator):
        super(Cifar10Eval, self).__init__()
        net = make_mnist(params['mnist_network_dims'])
        self.net = initialize_eval_mnist(net)


class AdversarialEval(Eval):
    def __init__(self):
        super(AdversarialEval, self).__init__()
        self.training_adversarial = flags['train_adversarial']
        self.img_no = 0
        if self.training_adversarial:
            self.classifier = Classifier(training=flags['classifier_train'])
            self.classifier.load()
    #Returns mean f of cluster (every input-output pair is considered)
    def _eval_single_cluster(self, real_data, reconstructions,  targets=None, predicted_logits=None):
        '''
        Args:
        predicted: logits from classifier
        '''

        self._visualize_img_pair((real_data[0], reconstructions[0]))
        dist = tf.norm(real_data - reconstructions, axis=1)

        if self.training_adversarial:
            ce  = tf.reduce_sum(- targets * (tf.math.log(tf.nn.softmax(predicted_logits) + EPS)), axis=1)
            return tf.reduce_mean(dist + ce)
        else:
            rate = tfd.kl_divergence(self.approx_posterior, self.latent_prior)
            return tf.reduce_mean(rate+dist)

    def _visualize_img_pair(self, pair):
        img_dims = params['img_dim']
        pair = [x.numpy().reshape(img_dims) for x in pair]
        img = tf.concat(pair, axis=0)
        if self.img_no % 50 == 0:
            try:
                plt.imshow(img)
                plt.savefig(str(self.img_no) + '.png')
            except:
                np.save(str(self.img_no), img)
        self.img_no += 1

    def _predict_cluster(self, cluster_items):
        logits, classes = self.classifier.classify(cluster_items)
        return logits, classes

    def set_latent_dists(self, latent_prior, approx_posterior):
        self.latent_prior = latent_prior
        self.approx_posterior = approx_posterior

    def eval_clusters(self, real, cluster_reconstruction_dict, targets=None):
        #TODO:
        # Check these during adversarial training and not
        # 1. feed correct target
        # 2. feed correct predicted
        self.clean_eval_memory()
        for cluster, values in cluster_reconstruction_dict.items():

            cluster_idx = [value [0] for value in values]

            cluster_real = tf.gather(real, cluster_idx)
            cluster_reconstruction = tf.stack([value[1] for value in values])
            cluster_targets = tf.gather(targets, cluster_idx)

            if self.training_adversarial:
                # handles adversarial training for adversarial application
                cluster_predicted_logits, cluster_classes = self._predict_cluster(cluster_reconstruction)
                print('Adversarial MNIST accuracy: {}'.format(self.classifier.accuracy(cluster_targets, cluster_classes)))
            else:
                # handles prior training for adversarial application
                cluster_predicted_logits = None
            self.metrics['f'].append(self._eval_single_cluster(cluster_real, cluster_reconstruction, cluster_targets, cluster_predicted_logits))

        self._tidy_metrics()
