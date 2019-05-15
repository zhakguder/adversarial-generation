import warnings
warnings.filterwarnings("ignore")
from data_2 import *
from estimators_2 import Generator, AdversarialGenerator, NetworkGenerator
from settings import get_settings
from data_generators import combined_data_generators
from applications import MnistEval, AdversarialEval
from utils import timeit_context
from sys import exit


flags, params = get_settings()
AUTOENCODE = flags['autoencode']
APP = flags['app']
CLASSIFIER = flags['classifier_train']
LOAD_CLASSIFIER = flags['load_classifier']
TRAIN_ADVERSARIAL = flags['train_adversarial']
ONLY_CLASSIFIER = flags['only_classifier']

EPOCHS = flags['epochs']


optimizer = tf.keras.optimizers.Adam()
loss_history = []

if ONLY_CLASSIFIER and CLASSIFIER: # train classifier for adversarial generation
    classifier = Classifier(training=CLASSIFIER)
    if LOAD_CLASSIFIER:
        classifier.load()
    for epoch in range(EPOCHS):
        x_train, y_train = classifier.get_input_data()
        x_test, y_test = classifier.get_input_data(train=False)
        with tf.GradientTape() as tape:
            logits = classifier(x_train)
            loss_value = tf.nn.softmax_cross_entropy_with_logits(y_train, logits)
            print('Mnist classifier loss: {}'.format(tf.reduce_mean(loss_value)))
            grads = tape.gradient(loss_value, classifier.trainable_variables)
            optimizer.apply_gradients(zip(grads, classifier.trainable_variables))
            _, predicted_classes = classifier.classify(x_test)
            print('Mnist test accuracy: {}'.format(classifier.accuracy(tf.argmax(y_test, axis=1), predicted_classes)))
    classifier.save()
    exit()

elif ONLY_CLASSIFIER and LOAD_CLASSIFIER and not CLASSIFIER:
    classifier = Classifier(training=CLASSIFIER)
    classifier.load()
    x_test, y_test = classifier.get_input_data(train=False)
    _, predicted_classes = classifier.classify(x_test)
    print('Mnist test accuracy: {}'.format(classifier.accuracy(tf.argmax(y_test, axis=1), predicted_classes)))
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
    evaluator = MnistEval(model.aux_data)

if flags['load_generator']:
    model.load()
# These are not epochs yet, but updates
for epoch in range(EPOCHS):
    print('Start of epoch %d' % (epoch,))

    # Iterate over the batches of the dataset.
    for step, data in enumerate(model.get_input_data()):
        try:
            x_train, y_train, y_logits  = data
        except:
            try:
                x_train, y_train = data
            except:
                x_train = data
        with tf.GradientTape() as tape:
            if APP =='generated':
                qs = model(x_train)
                means = model.get_cluster_centers()
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
                evaluator.eval_clusters(correct, model.cluster_dictionary, y_train)
                evaluator.summary()

            loss_value = tf.math.log(qs) + evaluator.metrics['f']

            model.loss.append(loss_value.numpy().mean())
            model.summary()

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % 100 == 0:
        model.save()
