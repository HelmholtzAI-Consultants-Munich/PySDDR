from sddr import SDDR
import argparse
import yaml


import torch.nn as nn
# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)

# see the readme for further explanation of the arguments
def get_args():
    '''
    Optional argument
    ------------------
        -c: The path to the config file of the test run
    '''
    parser = argparse.ArgumentParser(description='Predict heart volume for test image',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', '-c',
                        help='Give the configuration file with for the test run')
    return parser.parse_args()

    
if __name__ == '__main__':
    args = get_args()
    if args.config:
        # not yet implemented to read network architecture
        config = get_config(args.config) #load config file
        sddr = SDDR(config=config)
    else:
        # specify either a dict with params or a list of param values
        data_path = './example_data/simple_gam/X.csv'
        ground_truth_path = './example_data/simple_gam/Y.csv'
        output_dir = './outputs'

        current_distribution  = 'Poisson' #'Normal'
        '''
        formulas = {'loc': '~1+spline(x1, bs="bs",df=9)+spline(x2, bs="bs",df=9)+d1(x1)+d2(x2)',
                    'scale': '~1+spline(x2, bs="bs",df=9)'
                    }
        '''
        formulas = {'rate': '~1+spline(x1, bs="bs",df=9)+spline(x2, bs="bs",df=9)+d1(x1)+d2(x2)'}
        deep_models_dict = {
        'd1': {
            'model': nn.Sequential(nn.Linear(1,15)),
            'output_shape': 15},
        'd2': {
            'model': nn.Sequential(nn.Linear(1,3),nn.ReLU(), nn.Linear(3,8)),
            'output_shape': 8}
        }

        train_parameters = {
        'batch_size': 1000,
        'epochs': 2500,
        'regularization_params': {'rate': 1} #{'loc':1, 'scale':1}
        }

        sddr = SDDR(data_path=data_path,
                    ground_truth_path=ground_truth_path,
                    output_dir=output_dir,
                    current_distribution=current_distribution,
                    formulas=formulas,
                    deep_models_dict=deep_models_dict,
                    train_parameters=train_parameters)


    sddr.train()
    partial_effects = sddr.eval('rate')