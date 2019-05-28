import warnings
warnings.filterwarnings("ignore")
from data import *
from estimators import Generator, AdversarialGenerator, NetworkGenerator
from settings import get_settings
from data_generators import combined_data_generators
from applications import MnistEval, AdversarialEval, Cifar10Eval
from utils import timeit_context, min_max
from sys import exit
from ipdb import set_trace
from classifier import Classifier
import numpy as np

from empirical import patch_fitness_grad

EPS = tf.keras.backend.epsilon()

COMBINE_GRADS = False

flags, params = get_settings()
AUTOENCODE = flags['autoencode']
APP = flags['app']
CLASSIFIER = flags['classifier_train']
LOAD_CLASSIFIER = flags['load_classifier']
TRAIN_ADVERSARIAL = flags['train_adversarial']
ONLY_CLASSIFIER = flags['only_classifier']

LEARNING_RATE = params['learning_rate']

EPOCHS = flags['epochs']
N_DATA = 60000

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-5)
loss_history = []

if ONLY_CLASSIFIER and CLASSIFIER: # train classifier for adversarial generation
    classifier = Classifier(training=CLASSIFIER)
    if LOAD_CLASSIFIER:
        classifier.load()
    for epoch in range(EPOCHS):
        print('Starting epoch: {}'.format(epoch))
        for step, data in enumerate(classifier.get_input_data()):
            x_train, y_train = data
            if step > int(np.ceil(N_CIFAR10/x_train.shape[0])):
                break
            with tf.GradientTape() as tape:
                logits = classifier(x_train)
                loss_value = tf.nn.softmax_cross_entropy_with_logits(y_train, logits)
                if step % 20 == 0:
                    print('Step: {}'.format(step))
                    print('{} classifier loss: {}'.format(flags['dataset'], tf.reduce_mean(loss_value)))
                grads = tape.gradient(loss_value, classifier.trainable_variables)
                optimizer.apply_gradients(zip(grads, classifier.trainable_variables))
                for stept, test_data in enumerate(classifier.get_input_data(train=False)):
                    if stept ==  1:
                        break
                    x_test, y_test = test_data
                    _, predicted_classes = classifier.classify(x_test)
                    if step % 20 == 0:
                        classifier.accuracy(tf.argmax(y_test, axis=1), predicted_classes)
    classifier.save()
    exit()

elif ONLY_CLASSIFIER and LOAD_CLASSIFIER and not CLASSIFIER:
    classifier = Classifier(training=CLASSIFIER)
    classifier.load()
    for stept, test_data in enumerate(classifier.get_input_data(train=False)):
        if stept ==  1:
            break
    x_test, y_test = test_data
    _, predicted_classes = classifier.classify(x_test)
    classifier.accuracy(tf.argmax(y_test, axis=1), predicted_classes)
    exit()

elif AUTOENCODE:
    if APP == 'adversarial':
        # labels and logits predicted by a classifier
        model = AdversarialGenerator()
        evaluator = AdversarialEval()
    else:
        raise NotImplementedError ('No use case for autoencoding without adversarial generation.')
else:
    model = NetworkGenerator()
    model.get_input_data()
    evaluator = MnistEval(model.aux_data) if flags['dataset'] == 'mnist' else Cifar10Eval(model.aux_data)

if flags['load_generator']:
    model.load()

for epoch in range(EPOCHS):
    print('Start of epoch %d' % (epoch,))
    # Iterate over the batches of the dataset.
    for step, data in enumerate(model.get_input_data()):

        if type(data)==tuple:
            try:
                x_train, y_train, y_logits  = data
            except:
                x_train, y_train = data
        else:
            x_train = data

        if step > int(np.ceil(N_DATA/x_train.shape[0])):
            break

        print('Step: {}'.format(step))

        with tf.GradientTape(persistent=True) as tape:
            if APP =='generated':
                output = model(x_train)
                means = model.get_cluster_centers()
                with tf.GradientTape() as eval_tape:
                    evaluator.eval_clusters(means)
                evaluator.summary()

            elif APP == 'adversarial':
                # TODO
                # SINGLE IMAGE OR WHAT????
                # TODO:
                # 1. evaluate for each possible class when in adversarial training
                #else:

                qs = model((x_train, y_train))
                correct = tf.reshape(x_train, (-1, model.output_dim))
                if not TRAIN_ADVERSARIAL:
                    evaluator.set_latent_dists(model.latent_prior, model.approx_posterior)
                with tf.GradientTape() as eval_tape:
                    evaluator.eval_clusters(correct, model.cluster_dictionary, y_train)
                evaluator.summary()

            if COMBINE_GRADS:
                f_gradients = eval_tape.gradient(evaluator.metrics['f'], evaluator.net.trainable_variables)
                f_gradient = tf.concat([tf.reshape(v, (-1, )) for v in f_gradients], axis=0)
                f_gradient = tf.tile(tf.reshape(f_gradient, (1,-1)), [flags['latent_batch_size'],1])
                output1 = patch_fitness_grad(output, f_gradient)
                qs_1 = model.cluster_computations(output1)
                loss_value_1 = tf.math.log(qs_1+EPS) # q_s + fitness
                grads_1 = tape.gradient(loss_value_1, model.trainable_variables)
                loss_value, grads = loss_value_1, grads_1
            else:
                qs_0 = model.get_cluster_qs()
                loss_value_0 = tf.math.log(qs_0 + EPS) # only qs
                grads_0 = tape.gradient(loss_value_0, model.trainable_variables)
                loss_value, grads = loss_value_0, grads_0

            model.loss.append(loss_value.numpy().mean())
            model.summary()
        #min_max(grads, 'GRADIENTS')

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % 100 == 0:
        model.save()
