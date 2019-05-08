_params = {}
_params['hidden_dim'] = [2000,4000]
_params['latent_dim'] = 10
_params['batch_size'] = 2048
_params['mnist_batch_size'] = 2048
_params['n_samples'] = 1
_params['data_dir'] = "vae/data"
_params['learning_rate'] = 0.01
_params['max_steps'] = 200
_params['w'] = 4
_params['mnist_network_dims'] =  [10, 2, 5]

_flags = {
    'buffer_size': 10000,
    'batch_size': 2048,
    'mnist_batch_size': 1024,
    'num_epochs': 50,
    'input_dim_gen': 500
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
    #global params
    _params['network_out_dim'] = get_mnist_generator_output_dim()
    return _flags, _params
