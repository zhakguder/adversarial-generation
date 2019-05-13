import tensorflow as tf
from data_2 import get_dataset
from ipdb import set_trace

def _latent_generator(train):
    train_set, test_set = get_dataset(name='latent', adversarial=False, autoencode=False)
    dataset = train_set if train else test_set
    return dataset


def _data_generator(aux_dataset, train, adversarial,autoencode):
    train_set, test_set = get_dataset(name=aux_dataset, adversarial=adversarial, autoencode=autoencode)
    dataset = train_set if train else test_set
    return dataset

def combined_data_generators(flags, train=True):
    autoencode = flags['autoencode']
    adversarial = True if flags['app'] == 'adversarial' else False
    if autoencode:
        dataset = _data_generator(flags['dataset'], train, adversarial, autoencode)
        return dataset
    else:
        noise_dataset = _latent_generator(train)
        aux_dataset = _data_generator(flags['dataset'], train, adversarial, autoencode)
        return noise_dataset, aux_dataset
