from functools import reduce
from ipdb import set_trace


_flags = {
    'app':'generated',
    'train_adversarial': False,
    'autoencode': False,
    'dataset': 'mnist',
    'only_classifier': False,
    'classifier_train': False,
    'load_classifier': False,
    'load_generator': False,
    'load_checkpoint': False,
    'verbose': True,
    'timeExecution': True,
    'buffer_size': 70000,
    'data_batch_size': 128,
    'latent_batch_size': 128,
    'epochs': 1
}

DATASET = _flags['dataset']

_flags['classifier_path'] = DATASET + '_classifier.m'
_flags['generator_path'] = DATASET + '_generator.m'
_flags['checkpoint_path'] = DATASET +  '_generated_lsh_ckpt'

# For adversarial autoencoding
if DATASET == 'mnist':  # using dense
    IMG_DIM = (28, 28, 1)
    CLASSIFIER_N_CLASSES = 10
    OUTPUT_DIM = reduce(lambda x, y: x*y, IMG_DIM)
    CLASSIFIER_INPUT_DIM = IMG_DIM
elif DATASET == 'cifar10': # using Conv
    IMG_DIM = (32, 32, 3) # channels last
    CLASSIFIER_N_CLASSES = 10
    CLASSIFIER_INPUT_DIM = IMG_DIM
    OUTPUT_DIM = reduce(lambda x, y: x*y, IMG_DIM) #TODO change this after you have Conv generator to IMG_DIM

_params = {
    'hidden_dim': [200, 500, 1000],
    'latent_dim': 100,
    'latent_samples': 1,
    'data_dir': "vae/data",
    'learning_rate': 0.01,
    'max_steps': 200,
    'w': 4, #set to 10000 to get a single cluster for in adversarial application before adversarial training else 4
    'mnist_network_dims':  [10, 20, 30], #[100, 800, 300],
    'CNN_classifier_filters': [8, 16],#, 128],  # for network generation
    'CNN_classifier_dropout': [0.2, 0.3],#, 0.4], #for network generation
    'classifier_input_dim': CLASSIFIER_INPUT_DIM,
    'classifier_n_classes': CLASSIFIER_N_CLASSES,
    'img_dim': IMG_DIM,
}


def get_settings_():

    return _flags, _params

def get_settings():
    from generator_output_dims import set_output_dims
    flags, params = set_output_dims()
    return flags, params

if __name__ == '__main__':
    get_settings()
