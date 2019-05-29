from models import CNN_classifier_net, initialize_eval_network
from settings import get_settings_
from ipdb import set_trace

flags, params = get_settings_()

def set_output_dims():

    if flags['app'] == 'generated':
        params['network_out_dim'] = get_mnist_generator_output_dim() if flags['dataset'] == 'mnist' else get_cifar10_generator_output_dim()
    elif flags['app'] == 'adversarial':
        params['network_out_dim'] = OUTPUT_DIM #autoencode output dim
    return flags, params

def get_cifar10_generator_output_dim():
    prev_dim = params['img_dim']
    net = CNN_classifier_net (params['CNN_classifier_filters'], params['CNN_classifier_dropout'], prev_dim, params['classifier_n_classes'], False)
    net = initialize_eval_network(net)
    return net.count_params()

def get_mnist_generator_output_dim():
    #return net.count_params
    prev_dim = params['img_dim']
    net = CNN_classifier_net (params['CNN_classifier_filters'], params['CNN_classifier_dropout'], prev_dim, params['classifier_n_classes'], False)
    net = initialize_eval_network(net)
    print('Count: {}'.format(net.count_params()))
    return net.count_params()

    #prev_dim = params['classifier_input_dim']
    #out_dim = 0
    #for dim in params['mnist_network_dims']:
    #    out_dim += (prev_dim+1) * dim
    #    prev_dim = dim
    #dim = 10
    #out_dim += (prev_dim+1) * dim
    #return out_dim

if __name__ == '__main__':
    get_cifar10_generator_output_dim()
