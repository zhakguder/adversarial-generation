import tensorflow as tf
from data import get_dataset
from ipdb import set_trace

def _latent_generator(train):
    train_set, test_set = get_dataset(name='latent', adversarial=False)
    dataset = train_set if train else test_set
    return dataset


def _data_generator(aux_dataset, train, adversarial, train_adversarial):
    train_set, test_set = get_dataset(name=aux_dataset, adversarial=adversarial, adversarial_training=train_adversarial)
    dataset = train_set if train else test_set
    return dataset

def combined_data_generators(flags, train=True):
    adversarial = True if flags['app'] == 'adversarial' else False
    train_adversarial = flags['train_adversarial']
    if adversarial or flags['only_classifier']:
        dataset = _data_generator(flags['dataset'], train, adversarial, train_adversarial)
        return dataset
    else:
        noise_dataset = _latent_generator(train)
        aux_dataset = _data_generator(flags['dataset'], train, adversarial, train_adversarial)
        return noise_dataset, aux_dataset
