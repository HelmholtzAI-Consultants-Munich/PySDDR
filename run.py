from train import SDDR
from torch import nn

if __name__ == '__main__':
    # config = get_config() load config file
    config = {'data_path': r'./example_data/simple_gam/X.csv',
    'ground_truth_path': r'./example_data/simple_gam/Y.csv',

        'current_distribution': 'Poisson',
    'formulas': {'rate': '~1+spline(x1, bs="bs",df=9)+spline(x2, bs="bs",df=9)+d1(x1)+d2(x2)'},

    'deep_shapes': {'rate':{
      'd1': 15,
      'd2': 8}
    },
    'deep_models_dict': {'rate':{
      'd1': nn.Sequential(nn.Linear(1,15)),
      'd2': nn.Sequential(nn.Linear(1,3),nn.ReLU(), nn.Linear(3,8))}
    },

    'train_parameters': {
      'batch_size': 1000,
      'epochs': 2500,
      'regularization_params': {'rate':1}
    }}
    sddr = SDDR(config)
    sddr.train()