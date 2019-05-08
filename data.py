import tensorflow as tf
from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
n_train = x_train.shape[0]
n_test = x_test.shape[0]
x_train, x_test = x_train.reshape(n_train, -1) / 255.0, x_test.reshape(n_test, -1) / 255.0
