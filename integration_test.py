# import the sddr module
from sddr import SDDR
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import numpy as np
import torch
import os

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
    _, partial_effects_pred_rate = sddr.predict(data/2,clipping=True)
    
    #normalize partial effects and compare with ground truth
    x = partial_effects_pred_rate['rate'][0][0]
    y = normalize(partial_effects_pred_rate['rate'][0][1])

    y_target = normalize(x**2) # ground truth: quadratic effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.1, "Partial effect not properly estimated on unseen data in simple GAM."
    
    x = partial_effects_pred_rate['rate'][1][0]
    y = normalize(partial_effects_pred_rate['rate'][1][1])

    y_target = normalize(-x) # ground truth: linear effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.02, "Partial effect not properly estimated on unseen data in simple GAM."
    
    sddr.save('temp_simple_gam.pth')

    
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
    _, partial_effects = sddr.predict(data/2,clipping=True, plot=False) 
    partial_effects_pred_loc = partial_effects['loc'] 
    partial_effects_pred_scale = partial_effects['scale']

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


def integration_test_load_and_predict():
    '''
    Integration test of training and saving a GAMLSS model, then loading the same  model.
    The model is used to predict on unseen data right after training and once again after it
    is loaded. The results of the two must match for the test to pass.
    '''
    #set seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    #load data
    data_path = './example_data/gamlss/X.csv'
    target_path = './example_data/gamlss/Y.csv'

    data = pd.read_csv(data_path,delimiter=';')
    target = pd.read_csv(target_path)
    train_data = data.iloc[:800]
    train_target = data.iloc[:800]
    test_data = data.iloc[800:]

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
    sddr.train(target=train_target, structured_data=train_data)
    _, partial_effects = sddr.predict(test_data, clipping=True)
    sddr.save('temp_gamlss.pth')
    # load trained SDDR and predict
    pred_sddr = SDDR(output_dir=output_dir,
                distribution=distribution,
                formulas=formulas,
                deep_models_dict=deep_models_dict,
                train_parameters=train_parameters)
    pred_sddr.load('./outputs/temp_gamlss.pth', train_data)
    _, partial_effects_loaded = pred_sddr.predict(test_data, clipping=True)
    # compare partial effects
    for param in partial_effects.keys():
        for partial_effect, partial_effect_loaded in zip(partial_effects[param], partial_effects_loaded[param]):
            abs_err = (partial_effect[1] - partial_effect_loaded[1])
            assert sum(abs_err) < 0.001, "Partial effect not same with original prediction and prediction after load for param %s and partial effect %s"%(param, partial_effect[0])
    os.remove('./outputs/temp_gamlss.pth')


def integration_test_load_and_resume():
    '''
    Integration test of training and saving a GAMLSS model, then loading the same  model.
    The model is used to continue training.
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
        'epochs': 100,
        'degrees_of_freedom': {'rate': 6},
        'optimizer' : optim.RMSprop
    }
    #initialize SDDR
    sddr_100 = SDDR(output_dir=output_dir,
                distribution=distribution,
                formulas=formulas,
                deep_models_dict=deep_models_dict,
                train_parameters=train_parameters)

    # train SDDR
    sddr_100.train(target=target, structured_data=data)
    #_, partial_effects = sddr.predict(test_data, clipping=True)
    sddr_100.save('temp_simple_gam.pth')
    # load trained SDDR and predict
    train_parameters['epochs'] = 500
    sddr_resume = SDDR(output_dir=output_dir,
                distribution=distribution,
                formulas=formulas,
                deep_models_dict=deep_models_dict,
                train_parameters=train_parameters)
    sddr_resume.load('./outputs/temp_simple_gam.pth', data)
    sddr_resume.train(target=target, structured_data=data, resume=True)
    loss_resume = sddr_resume.epoch_loss
    # train continuously
    sddr_500 = SDDR(output_dir=output_dir,
            distribution=distribution,
            formulas=formulas,
            deep_models_dict=deep_models_dict,
            train_parameters=train_parameters)
    sddr_500.train(target=target, structured_data=data)
    loss_500 = sddr_500.epoch_loss
    loss_dif = abs(loss_500 - loss_resume)
    assert loss_dif < 0.001, "Loss function not equal in two training methods"

    os.remove('./outputs/temp_simple_gam.pth')
        
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

    print("Test loading a GAMLSS model and predicting")
    integration_test_load_and_predict()
    print("-----------------------")
    print("passed tests for loading and predicting")

    print("Test loading a GAMLSS model and resuming training")
    integration_test_load_and_resume()
    print("-----------------------")
    print("passed tests for loading and resuming training")