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
    IMG_DIM = (28, 28)
    CLASSIFIER_N_CLASSES = 10
    OUTPUT_DIM =  CLASSIFIER_INPUT_DIM = reduce(lambda x, y: x*y, IMG_DIM)
elif DATASET == 'cifar10': # using Conv
    IMG_DIM = (32, 32, 3) # channels last
    CLASSIFIER_N_CLASSES = 10
    CLASSIFIER_INPUT_DIM = IMG_DIM
    OUTPUT_DIM = reduce(lambda x, y: x*y, IMG_DIM) #TODO change this after you have Conv generator to IMG_DIM

_params = {
    'hidden_dim': [100, 200, 500],
    'latent_dim': 20,
    'latent_samples': 1,
    'data_dir': "vae/data",
    'learning_rate': 0.01,
    'max_steps': 200,
    'w': 4, #set to 10000 to get a single cluster for in adversarial application before adversarial training else 4
    'mnist_network_dims':  [10, 20, 30], #[100, 800, 300],
    'classifier_input_dim': CLASSIFIER_INPUT_DIM,
    'classifier_n_classes': CLASSIFIER_N_CLASSES,
    'img_dim': IMG_DIM,
}

def get_mnist_generator_output_dim():
    prev_dim = 784
    out_dim = 0
    for dim in _params['mnist_network_dims']:
        out_dim += (prev_dim+1) * dim
        prev_dim = dim
    dim = 10
    out_dim += (prev_dim+1) * dim
    return out_dim
def get_settings():
    if _flags['app'] == 'generated':
        _params['network_out_dim'] = get_mnist_generator_output_dim()
    elif _flags['app'] == 'adversarial':
        _params['network_out_dim'] = OUTPUT_DIM #autoencode output dim

    return _flags, _params
