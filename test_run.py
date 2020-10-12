from sddr import SDDR
import argparse
import yaml

import torch.optim as optim
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
        config = get_config(args.config) #load config file
        sddr = SDDR(config=config)
    else:
        # specify either a dict with params or a list of param values
        data = './example_data/simple_gam/X.csv'
        target = './example_data/simple_gam/Y.csv'
        output_dir = './outputs'
        mode = 'train'
        resume = None # ./outputs/model.pth

        distribution  = 'Poisson'

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
        'epochs': 1000,
        'optimizer': optim.SGD,
        'optimizer_params':{'lr': 0.01, 'momentum': 0.9}, 
        'regularization_params': {'rate': 1}
        }

    sddr = SDDR(data=data,
                mode=mode,
                target=target,
                output_dir=output_dir,
                distribution=distribution,
                formulas=formulas,
                deep_models_dict=deep_models_dict,
                train_parameters=train_parameters)

    
    if resume in locals() and resume:
        sddr.load(resume)
    sddr.train(plot=True)
    partial_effects = sddr.eval('rate')
    sddr.save('model.pth')

