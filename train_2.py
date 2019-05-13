import warnings
warnings.filterwarnings("ignore")
from data_2 import *
from estimators_2 import Generator, AdversarialGenerator, NetworkGenerator
from settings import get_settings
from data_generators import combined_data_generators
from applications import MnistEval, AdversarialEval
from utils import timeit_context


flags, params = get_settings()
AUTOENCODE = flags['autoencode']
APP = flags['app']

EPOCHS = flags['epochs']

#model = initialize_model(Generator, train_data, params, flags)
optimizer = tf.keras.optimizers.Adam()

# calculate and update loss
loss_history = []

if AUTOENCODE:
    if APP == 'adversarial':
        # labels and logits predicted by a classifier
        model = AdversarialGenerator(params, flags)
        data = model.get_input_data()
        x_train, y_train, y_logits  = data
        evaluator = AdversarialEval()
    else:
        raise NotImplementedError ('No use case for autoencoding without adversarial generation.')
else:
    model = NetworkGenerator(params, flags)
    data = model.get_input_data()
    x_train = data
    evaluator = MnistEval(model.aux_data)

# These are not epochs yet, but updates
for epoch in range(EPOCHS):
    with timeit_context('One batch:'):
        with tf.GradientTape() as tape:
            if APP =='generated':
                with timeit_context('Calc qs'):
                    qs = model(x_train)
                means = model.get_cluster_centers()
                with timeit_context('Cluster performance'):
                    evaluator.eval_clusters(means)
                evaluator.summary()

            elif APP == 'adversarial':
                # TODO
                # SINGLE IMAGE OR WHAT????
                with timeit_context('Calc qs'):
                    qs = model((x_train, y_train))
                correct = tf.reshape(x_train, (-1, model.output_dim))
                # TODO:
                # 1. call classifier to get predicted (y_logits is a substitute for now)
                # 2. evaluate for each possible class (tf.one_hot(3, 10) is a substitute for now)
                with timeit_context('Cluster performance'):
                    evaluator.eval_clusters(correct, model.cluster_dictionary, tf.one_hot(3, 10), y_logits[0])
                evaluator.summary()

            loss_value = tf.math.log(qs) + evaluator.metrics['f']

            model.loss.append(loss_value.numpy().mean())
            model.summary()

            with timeit_context('Calc grads'):
                grads = tape.gradient(loss_value, model.trainable_variables)
            with timeit_context('Apply grads'):
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
