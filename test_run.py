from sddr import SDDR
import argparse
import yaml

import torch.optim as optim
import torch.nn as nn

#from model import TestNet
#import torchvision.models as models

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
    parser = argparse.ArgumentParser(description='Run pySDDR',
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
        structured_data = './example_data/simple_gam/X.csv'
        unstructured_data = {
            'x3':{
                'path': './mnist_data/mnist_images',
                'datatype': 'image'
            }
        } 
        
        target = './example_data/simple_gam/Y.csv'
        output_dir = './outputs'
        mode = 'train'
        load_model = None # ./outputs/model.pth

        distribution  = 'Poisson'

        formulas = {'rate': '~1+spline(x1, bs="bs",df=9)+spline(x2, bs="bs",df=9)+d1(x1)+d2(x2)'}
        deep_models_dict = {
        'd1': {
            'model': nn.Sequential(nn.Linear(1,3),nn.ReLU(), nn.Linear(3,4)),
            'output_shape': 4},

        'd2': {
            'model': nn.Sequential(nn.Linear(1,3),nn.ReLU(), nn.Linear(3,4)),
            'output_shape': 4}
        }

        train_parameters = {
        'batch_size': 200,
        'epochs': 200,
        'optimizer': optim.SGD,
        'optimizer_params':{'lr': 0.01, 'momentum': 0.9}, 
        'degrees_of_freedom': {'rate': 10}
        }

        sddr = SDDR(structured_data=structured_data,
                    mode=mode,
                    target=target,
                    output_dir=output_dir,
                    distribution=distribution,
                    formulas=formulas,
                    deep_models_dict=deep_models_dict,
                    train_parameters=train_parameters,
                    unstructured_data=unstructured_data)

    
    is_local = "load_model" in locals()
    if is_local:
        if load_model :
            sddr.load()
    elif args.config:
        if config['load_model']:
            sddr.load()
    sddr.train(plot=True)
    #partial_effects = sddr.eval('rate')
    sddr.save('model.pth')

    '''
    'model': nn.Sequential(nn.Flatten(1, -1),
                nn.Linear(28*28,256),
                nn.ReLU(),
                nn.Linear(256,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Linear(64,10)),
    'output_shape': 10}, #1000 for alexnet
    '''