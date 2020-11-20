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

def integration_test_simple_gam():
    '''
    Integration test using a Simple GAM Poisson Distribution.
    The partial effects are estimated and compared with the ground truth 
    (only functional form: the terms are normalized before comparison)
    If the error is higher than a resonable value an error is raised.
    '''
    #set seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    #load data
    data_path = './example_data/simple_gam/X.csv'
    target_path = './example_data/simple_gam/Y.csv'

    data = pd.read_csv(data_path,delimiter=';')
    target = pd.read_csv(target_path)

    output_dir = './outputs'

    #define SDDR parameters
    distribution  = 'Poisson'

    formulas = {'rate': '~1 + spline(x1, bs="bs",df=9) + spline(x2, bs="bs",df=9) + d1(x1) + d2(x2)'}
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
        'epochs': 500,
        'degrees_of_freedom': {'rate': 6},
        'optimizer' : optim.RMSprop
    }

    #initialize SDDR
    sddr = SDDR(output_dir=output_dir,
                distribution=distribution,
                formulas=formulas,
                deep_models_dict=deep_models_dict,
                train_parameters=train_parameters)
    
    # train SDDR
    sddr.train(target=target, structured_data=data)
    
    #compute partial effects
    partial_effects_rate = sddr.eval('rate',plot=False)

    #normalize partial effects and compare with ground truth
    x = partial_effects_rate[0][0]
    y = normalize(partial_effects_rate[0][1])

    y_target = normalize(x**2) # ground truth: quadratic effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.1, "Partial effect not properly estimated in simple GAM."
    
    x = partial_effects_rate[1][0]
    y = normalize(partial_effects_rate[1][1])

    y_target = normalize(-x) # ground truth: linear effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.02, "Partial effect not properly estimated in simple GAM."
    
    #compute partial effects on unseen data
    _, partial_effects_pred_rate = sddr.predict(data/2,clipping=True,param='rate')
    
    #normalize partial effects and compare with ground truth
    x = partial_effects_pred_rate[0][0]
    y = normalize(partial_effects_pred_rate[0][1])

    y_target = normalize(x**2) # ground truth: quadratic effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.1, "Partial effect not properly estimated on unseen data in simple GAM."
    
    x = partial_effects_pred_rate[1][0]
    y = normalize(partial_effects_pred_rate[1][1])

    y_target = normalize(-x) # ground truth: linear effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.02, "Partial effect not properly estimated on unseen data in simple GAM."
    

    
def integration_test_gamlss():
    '''
    Integration test using a GAMLSS - Logistic Distribution.
    The partial effects are estimated and compared with the ground truth 
    (only functional form: the terms are normalized before comparison)
    If the error is higher than a resonable value an error is raised.
    '''
    #set seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    #load data
    data_path = './example_data/gamlss/X.csv'
    target_path = './example_data/gamlss/Y.csv'

    data = pd.read_csv(data_path,delimiter=';')
    target = pd.read_csv(target_path)

    output_dir = './outputs'

    #define SDDR parameters
    distribution  = 'Logistic'

    formulas = {'loc': '~1+spline(x1, bs="bs", df=4)+spline(x2, bs="bs",df=4) + d1(x1)+d2(x2)',
                'scale': '~1 + spline(x3, bs="bs",df=4) + spline(x4, bs="bs",df=4)'
                }

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
        'epochs': 200,
        'degrees_of_freedom': {'loc':4, 'scale':4},
        'optimizer' : optim.RMSprop
    }

    #initialize SDDR
    sddr = SDDR(output_dir=output_dir,
                distribution=distribution,
                formulas=formulas,
                deep_models_dict=deep_models_dict,
                train_parameters=train_parameters)
    
    # train SDDR
    sddr.train(target=target, structured_data=data)
    
    #compute partial effects
    partial_effects_loc = sddr.eval('loc',plot=False)
    partial_effects_scale = sddr.eval('scale',plot=False)

    #normalize partial effects and compare with ground truth
    x = partial_effects_loc[0][0]
    y = normalize(partial_effects_loc[0][1])

    y_target = normalize(x**2) # ground truth: quadratic effect

    RMSE = (y-y_target).std()

    assert RMSE<0.12, "Partial effect not properly estimated in GAMLSS."
    
    x = partial_effects_loc[1][0]
    y = normalize(partial_effects_loc[1][1])

    y_target = normalize(-x) # ground truth: linear effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.1, "Partial effect not properly estimated in GAMLSS."
    
    x = partial_effects_scale[0][0]
    y = normalize(partial_effects_scale[0][1])

    y_target = normalize(x) # ground truth: linear effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.15, "Partial effect not properly estimated in GAMLSS."
    
    x = partial_effects_scale[1][0]
    y = normalize(partial_effects_scale[1][1])

    y_target = normalize(np.sin(4*x)) # ground truth: sinusoidal effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.4, "Partial effect not properly estimated in GAMLSS."
        
    #compute partial effects on unseen data
    _, partial_effects_pred_loc = sddr.predict(data/2,clipping=True,param='loc', plot=False) 
    _, partial_effects_pred_scale = sddr.predict(data/2,clipping=True,param='scale', plot=False)

    #normalize partial effects and compare with ground truth
    x = partial_effects_pred_loc[0][0]
    y = normalize(partial_effects_pred_loc[0][1])

    y_target = normalize(x**2) # ground truth: quadratic effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.25, "Partial effect not properly estimated in GAMLSS."
    
    x = partial_effects_pred_loc[1][0]
    y = normalize(partial_effects_pred_loc[1][1])

    y_target = normalize(-x) # ground truth: linear effect

    RMSE = (y-y_target).std()

    assert RMSE<0.1, "Partial effect not properly estimated in GAMLSS."
    
    x = partial_effects_pred_scale[0][0]
    y = normalize(partial_effects_pred_scale[0][1])

    y_target = normalize(x) # ground truth: linear effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.15, "Partial effect not properly estimated in GAMLSS."
    
    x = partial_effects_pred_scale[1][0]
    y = normalize(partial_effects_pred_scale[1][1])

    y_target = normalize(np.sin(4*x)) # ground truth: sinusoidal effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.4, "Partial effect not properly estimated in GAMLSS."
  
        
    
def integration_test_mnist():
    '''
    Integration test with unstructed data.
    A mixed model is used that has structued and unstructured input.
    The unstructured input are mnist images. These mnist images are used in the formula and represent the number that is
    on the image. The test tests if the estimated numbers are on average (median) a monotonically increasing function of the
    true numbers on the mnist images
    '''


    #set seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)

    #load data
    data_path = './mnist_data/tab.csv'

    data = pd.read_csv(data_path,delimiter=',').loc[:1000,:]

    for i in data.index:

        data.loc[i,'groundtruth'] = np.sin(data.loc[i,'x1']) - 3*data.loc[i,'x2'] + data.loc[i,'x3']**4 + 3*data.loc[i,'y_true']


    data.loc[:,'groundtruth'] = data.loc[:,'groundtruth'] - data.loc[:,'groundtruth'].mean()

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
        'batch_size': 100,
        'epochs': 100,
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
               target="groundtruth",
               unstructured_data = unstructured_data,
              plot=True)

    data_pred = data.loc[:,:]
    distribution_layer, partial_effect = sddr.predict(data_pred,
                                                      clipping=True,
                                                      param='scale', 
                                                      plot=False, 
                                                      unstructured_data = unstructured_data)

    assert distribution_layer.scale[0]>0.7, "Scale too large in mnist test"

    data_pred2 = data.copy()

    data_pred2.loc[:,'x1'] = 0
    data_pred2.loc[:,'x2'] = 0
    data_pred2.loc[:,'x3'] = 0
    data_pred2

    distribution_layer, partial_effect = sddr.predict(data_pred2,
                                                      clipping=True,
                                                      param='scale', 
                                                      plot=False, 
                                                      unstructured_data = unstructured_data)

    data_pred2['predicted_number'] = distribution_layer.loc[:,:].numpy().flatten()


    predicted_numbers  = data_pred2.groupby('y_true').median().predicted_number
    maximum_deviation_mnist = abs((predicted_numbers.loc[1:].to_numpy() - predicted_numbers.loc[:8].to_numpy())/3 - 1).max()

    assert distribution_layer.scale[0]>1, "Predicted numbers for the mnist not monotonically increasing"
    
    
if __name__ == '__main__':
    
    # run integration tests
    print("Test simple GAM")
    integration_test_simple_gam()  
    print("---------------------------")
    print("passed tests for simple GAM")
    
    print("Test simple GAMLSS")
    integration_test_gamlss()   
    print("-----------------------")
    print("passed tests for GAMLSS")
    
    print("Test with MNIST data")
    integration_test_mnist()   
    print("-----------------------")
    print("passed tests for MNIST data")