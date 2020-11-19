'''
UNDER DEVELOPMENT
'''


# import the sddr module
from sddr import SDDR
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import numpy as np
import torch


def normalize(x):
    x = x - x.mean()
    x = x/x.std()
    return x



#set seeds for reproducibility
torch.manual_seed(1)
np.random.seed(1)

#load data
data_path = './mnist_data/tab.csv'

data = pd.read_csv(data_path,delimiter=',')

output_dir = './outputs'

unstructured_data = {
  'numbers' : {
    'path' : './mnist_data/mnist_images',
    'datatype' : 'image'
  }
}

for i in data.index:
    data.loc[i,'numbers'] = f'img_{i}.jpg'

#define SDDR parameters
formulas = {'loc': '~ -1 + spline(x1, bs="bs", df=10) + x2 + dnn(numbers) + spline(x3, bs="bs", df=10)',
            'scale': '~1'
            }
distribution  = 'Normal'

deep_models_dict = {
'dnn': {
    'model': nn.Sequential(nn.Flatten(1, -1),
                           nn.Linear(28*28,128),
                           nn.ReLU()),
    'output_shape': 128},
}

train_parameters = {
    'batch_size': 8000,
    'epochs': 10,
    'degrees_of_freedom': {'loc':9.6, 'scale':9.6},
    'optimizer' : optim.RMSprop
}

#initialize SDDR
sddr = SDDR(output_dir=output_dir,
            distribution=distribution,
            formulas=formulas,
            deep_models_dict=deep_models_dict,
            train_parameters=train_parameters,
            )

# train SDDR
sddr.train(structured_data=data,
           target="y_gen",
           unstructured_data = unstructured_data,
          plot=True)

#compute partial effects
partial_effects_loc = sddr.eval('loc',plot=True)
partial_effects_scale = sddr.eval('scale',plot=True)

data_pred = data.loc[:10,:]
distribution_layer, partial_effect = sddr.predict(data_pred,
                                                  clipping=True,
                                                  param='scale', 
                                                  plot=False, 
                                                  unstructred_data_info = unstructured_data)

print(distribution_layer.scale)