import tensorflow as tf
from settings import get_settings
from tensorflow.keras import Model, models
from models import mnist_classifier_net, cifar10_classifier_net

from ipdb import set_trace

flags, params = get_settings()


class Classifier(Model):
    def __init__(self, training=True):
        super(Classifier, self).__init__()
        from data_generators import combined_data_generators
        self.mnist_input_shape_ = (params['classifier_input_dim'],)
        self.cifar10_input_shape_ = params['classifier_input_dim']
        self.output_shape_ = params['classifier_n_classes']
        if flags['classifier_train']:
            flags['train_adversarial'] = False
            flags['autoencode'] = False
        self.input_data = combined_data_generators(flags)
        self.test_data = combined_data_generators(flags, train=False)
        self.net = mnist_classifier_net(self.mnist_input_shape_, self.output_shape_, training) if flags['dataset'] == 'mnist' else cifar10_classifier_net([32, 64, 128], [0.2, 0.3, 0.4], self.cifar10_input_shape_, self.output_shape_, training)

        self._accuracy = tf.keras.metrics.Accuracy()
        self.model_path = flags['classifier_path']
        print('In classifier: {}'.format('training' if training else 'not training'))
    def get_input_data(self, train=True):
        return self.input_data if train else self.test_data
    def call(self, inputs):
        return self.net(inputs)
    def accuracy(self, labels, pred_classes):
        labels_preds = {}
        for k, v in zip(['y_true', 'y_pred'], [labels, pred_classes]):
            try:
                v = tf.argmax(v, axis=1)
            except:
                pass
            labels_preds[k] = v
        self._accuracy.update_state(**labels_preds)
        result = self._accuracy.result()
        print('{} classifier accuracy: {}'.format(flags['dataset'], result))
        return result

    def classify(self, x_eval):
        predicted_logits = self.net(x_eval)
        predicted_classes = tf.argmax(tf.nn.softmax(predicted_logits), axis=1)
        return predicted_logits, predicted_classes

    def save(self):
        self.net.save(self.model_path)
    def load(self):
        self.net = models.load_model(self.model_path)
        print('Classifier loaded!!')


if __name__ == "__main__":

    from tensorflow.keras.datasets import cifar10
    from ipdb import set_trace
    (_,_),(x_test, y_test) = cifar10.load_data()
    x_test = tf.cast(x_test[:1000]/255., tf.float32)
    classifier = Classifier(False)
    classifier.load()
    set_trace()
    predicted_logits, predicted_classes = classifier.classify(x_test)
