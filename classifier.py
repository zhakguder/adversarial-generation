from sklearn.linear_model import LogisticRegression
from tensorflow.keras.datasets import mnist
from ipdb import set_trace

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train[:100, :, :] / 255.0, x_test / 255.0
n_samples = x_train.shape[0]
y_train = y_train[:n_samples]

n_test_samples = x_test.shape[0]
classifier = LogisticRegression(random_state=0, solver='lbfgs', max_iter = 300,  multi_class='multinomial').fit(x_train.reshape(n_samples,  -1), y_train)

res = classifier.predict(x_test.reshape(n_test_samples, -1))
