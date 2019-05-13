import tensorflow as tf

class Classifier:
    def __init__(self, data):
        '''
        Args:
        data: (x_train, y_train)
        '''
        self.x_train, self.y_train = data
        n_classes = tf.reduce_max(self.y_train)+1
        self.input_shape = self.x_train.shape[1:]
        self.output_shape = n_classes

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=self.input_shape, trainable=False),
            tf.keras.layers.Dense(128, activation='relu', trainable=False),
            tf.keras.layers.Dense(self.output_shape, trainable=False)
            ])
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    def classify(self, data):
        predicted_logits = self.model(data)
        predicted_classes = tf.argmax(tf.nn.softmax(predicted_logits), axis=1)
        return predicted_logits, predicted_classes


if __name__ == "__main__":

    from tensorflow.keras.datasets import mnist
    from ipdb import set_trace
    (train_data),(_, _) = mnist.load_data()

    classifier = Classifier(train_data)
    predicted_logits, predicted_classes = classifier.classify()
