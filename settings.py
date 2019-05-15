from functools import reduce
from ipdb import set_trace

_params = {}
_params['hidden_dim'] = [2000,4000]
_params['latent_dim'] = 100
_params['mnist_batch_size'] = 256
_params['latent_samples'] = 1
_params['data_dir'] = "vae/data"
_params['learning_rate'] = 0.01
_params['max_steps'] = 200
_params['w'] = 100000 #set to 10000 to get a single cluster for in adversarial application before adversarial training else 4
_params['mnist_network_dims'] =  [10, 20, 30]

_flags = {
    'buffer_size': 70000,
    'latent_batch_size': 2048,
    'data_batch_size': 256,
    'input_dim_gen': 500,
    'app':'adversarial',
    'only_classifier': True,
    'train_adversarial': False,
    'dataset': 'cifar10',
    'autoencode': False,
    'epochs': 2000,
    'verbose': True,
    'timeExecution': True,
    'classifier_train': True,
    'classifier_path': 'cifar10_classifier.m',
    'generator_path': 'generator.m',
    'load_classifier': False,
    'load_generator': False,
    'checkpoint_path': 'adversarial_lsh_ckpt',
    'load_checkpoint': False
}

# For adversarial autoencoding
DATASET = _flags['dataset']
if DATASET == 'mnist':  # using dense
    IMG_DIM = (28, 28)
    CLASSIFIER_N_CLASSES = 10
    OUTPUT_DIM =  CLASSIFIER_INPUT_DIM = reduce(lambda x, y: x*y, IMG_DIM)
elif DATASET == 'cifar10': # using Conv
    IMG_DIM = (32, 32, 3) # channels last
    CLASSIFIER_N_CLASSES = 10
    CLASSIFIER_INPUT_DIM = IMG_DIM
    OUTPUT_DIM = reduce(lambda x, y: x*y, IMG_DIM) #TODO change this after you have Conv generator to IMG_DIM

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
    else:
        _params['network_out_dim'] = OUTPUT_DIM #autoencode output dim
        _params['classifier_input_dim'] = CLASSIFIER_INPUT_DIM
        _params['classifier_n_classes'] = CLASSIFIER_N_CLASSES
        _params['img_dim'] = IMG_DIM
    return _flags, _params
