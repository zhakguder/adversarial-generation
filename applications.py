import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from ipdb import set_trace
from models import make_mnist, initialize_eval_mnist, set_mnist_weights
from settings import *

_, params = get_settings()

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
    def clean_eval_memory(self):
        self.metrics = {k: [] for k,v in self.metrics.items()}
    def summary(self):
        for k, v in self.get_cluster_metrics().items():
            if len(v) > 0:
                print('{}: {}'.format(k,v ))


class MnistEval(Eval):
    def __init__(self, data_generator):
        super(MnistEval, self).__init__()
        self._data = data_generator
        net = make_mnist(params['mnist_network_dims'])
        self.net = initialize_eval_mnist(net)
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
        n_clusters = cluster_means.shape[0]
        for i in range(n_clusters):
            images, labels = self._get_data()
            self.metrics['f'].append(self._eval_single_cluster(cluster_means[i], images, labels))
        self.metrics = {k: tf.stack(v) for k,v in self.metrics.items()}

class AdversarialEval(Eval):
    def __init__(self):
        super(AdversarialEval, self).__init__()

    #Returns mean f of cluster (every input-output pair is considered)
    def _eval_single_cluster(self, real_data, reconstruction,  target, predicted):
        dist = tf.norm(real_data - reconstruction, axis=1)
        ce  = tf.reduce_sum(- target * (tf.math.log(tf.nn.softmax(predicted) + EPS)), axis=1)
        return tf.reduce_mean(dist + ce)

    # dummy
    def _set_dummy_target_predicted(self, n_items):
        t_p = {}
        for item in ['target', 'predicted']:
            tmp = [int(x) for x in tf.floor(tf.random.uniform((n_items,)) * 10).numpy()]
            tmp_tensor = tf.one_hot(tmp, 10)
            t_p[item] = tmp_tensor
        return t_p

    def eval_clusters(self, real, cluster_reconstruction_dict, target, predicted):
        #TODO:
        # 1. feed correct target
        # 2. feed correct predicted
        for cluster, values in cluster_reconstruction_dict.items():

            cluster_idx = [value [0] for value in values]

            n_items = len(cluster_idx) #TODO: delete after todo item 1
            t_p = self._set_dummy_target_predicted(n_items)#TODO: delete after todo item 1
            #TODO: delete all occurences of t_p

            cluster_real = tf.gather(real, cluster_idx)
            cluster_reconstruction = tf.stack([value[1] for value in values])
            self.metrics['f'].append(self._eval_single_cluster(cluster_real, cluster_reconstruction, t_p['target'], t_p['predicted']))

#    def eval_cluster(self, cluster):
