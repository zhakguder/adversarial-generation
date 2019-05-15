import tensorflow as tf
from settings import get_settings
from tensorflow.keras import Model, models

from ipdb import set_trace

flags, params = get_settings()


class Classifier(Model):
    def __init__(self, training=True):
        super(Classifier, self).__init__()
        from data_generators import combined_data_generators
        self.input_shape_ = (params['classifier_input_dim'],)
        self.output_shape_ = params['classifier_n_classes']
        if flags['classifier_train']:
            flags['train_adversarial'] = False
            flags['autoencode'] = False
            self.input_data = combined_data_generators(flags)
            self.test_data = combined_data_generators(flags, train=False)
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=self.input_shape_),
            tf.keras.layers.Dense(128, activation='relu', trainable=training),
            tf.keras.layers.Dense(256, activation='relu', trainable=training),
            tf.keras.layers.Dense(self.output_shape_, trainable=training)
            ])
        self._accuracy = tf.keras.metrics.Accuracy()
        self.model_path = flags['classifier_path']
        print('In classifier: {}'.format('training' if training else 'not training'))
    def get_input_data(self, train=True):
        dataset = self.input_data if train else self.test_data
        for data in dataset.take(1):
            return data
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
        return self._accuracy.result()

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

    from tensorflow.keras.datasets import mnist
    from ipdb import set_trace
    (train_data),(_, _) = mnist.load_data()

    classifier = Classifier(train_data)
    predicted_logits, predicted_classes = classifier.classify()
