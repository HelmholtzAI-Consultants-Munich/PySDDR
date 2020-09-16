import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd


## SDDR NETWORK PART
class Sddr_Single(nn.Module):
    '''
    This class represents an sddr network with a structured part, one or many deep models, a linear layer for 
    the structured part and a linear layer for the concatenated outputs of the deep models. The concatenated 
    outputs of the deep models are first filtered with an orthogonilization layer which removes any linear 
    parts of the deep output (by taking the Q matrix of the QR decomposition of the output of the structured part). 
    The two outputs of the linear layers are added so a prediction of a single parameter of the distribution is made
    and is returned as the final output of the network.
    The model follows the architecture depicted here:
    https://docs.google.com/presentation/d/1cBgh9LoMNAvOXo2N5t6xEp9dfETrWUvtXsoSBkgVrG4/edit#slide=id.g8ed34c120e_0_0

    Parameters
    ----------
    deep_models_dict: dict
        dictionary where keys are names of the deep models and values are objects that define the deep models
    deep_shapes: dict
        dictionary where keys are names of the deep models and values are the outputs shapes of the deep models
    struct_shapes: int?
        number of structural features
    P: numpy matrix 
        matrix used for the smoothing regularization (with added zero matrix in the beginning for the linear part)
    Attributes
    ----------
    P: numpy matrix 
        matrix used for the smoothing regularization (with added zero matrix in the beginning for the linear part)
    deep_models_dict: dict
        dictionary where keys are names of the deep models and values are objects that define the deep models
    structured_head: nn.Linear
        A linear layer which is fed the structured part of the data
    deep_head: nn.Linear
        A linear layer which is fed the unstructured part of the data
    deep_models_exist: Boolean
        This value is true if deep models have been used on init of the ssdr_single network, otherwise it is false
    '''
    
    def __init__(self, deep_models_dict, deep_shapes, struct_shapes, P):
        
        super(Sddr_Single, self).__init__()
        self.P = P
        self.deep_models_dict = deep_models_dict
        
        #register external neural networks
        for key, value in deep_models_dict.items():
            self.add_module(key,value)
        
        
        self.structured_head = nn.Linear(struct_shapes,1, bias = False)
        
        if len(deep_models_dict) != 0:
            output_size_of_deep_models  = sum([deep_shapes[key] for key in deep_shapes.keys()])
            self.deep_head = nn.Linear(output_size_of_deep_models,1, bias = False)
            self._deep_models_exist = True
        else:
            self._deep_models_exist = False
        
              
        
    def _orthog_layer(self, Q, Uhat):
        """
        Utilde = Uhat - QQTUhat
        """
        Projection_Matrix = Q @ Q.T
        Utilde = Uhat - Projection_Matrix @ Uhat
        
        return Utilde
    
    
    def forward(self, datadict):
        """Comment 6.8.2020 We checked that we can actually have a dictionary as an input here. that should work fine"""
        # check if dataframe in structired is empty!!
        X = datadict["structured"]
        
        if self._deep_models_exist:
            Q, R = torch.qr(X)

            Uhat_list = []
            for key in self.deep_models_dict.keys(): #assume that the input for the NN has the name of the NN as key
                net = self.deep_models_dict[key]
                Uhat_list.append(net(datadict[key]))
            
            Uhat = torch.cat(Uhat_list, dim = 1) #concatenate the outputs of the deep NNs

            Utilde = self._orthog_layer(Q, Uhat)
            
            deep_pred = self.deep_head(Utilde)
        else:
            deep_pred = 0
        
        structured_pred = self.structured_head(X)
        
        pred = structured_pred + deep_pred

        return pred
    
    def get_regularization(self):
        P = torch.from_numpy(self.P).float() # should have shape struct_shapes x struct_shapes, numpy array
        weights = self.structured_head.weight #should have shape 1 x struct_shapes
        
        
        regularization = weights @ P @ weights.T
        
        return regularization
        
        
        
class Sddr(nn.Module):
    '''
    This class represents the full sddr network which can consist of one or many smaller sddr nets (in a parallel manner).
    Each smaller sddr predicts one distribution parameter and these are then sent into a transformation layer which applies
    constraints on the parameters depending on the given distribution. The output parameters are then fed into a distributional
    layer and a log-loss is computed. A regularization term is added to the log-loss to compute the total loss of the network.
    The model follows the architecture depicted here:
    https://docs.google.com/presentation/d/1cBgh9LoMNAvOXo2N5t6xEp9dfETrWUvtXsoSBkgVrG4/edit#slide=id.g8ed34c120e_5_16

    Parameters
    ----------
        family: string 
            A string describing the given distribution, e.g. "gaussian", "binomial", ...
        regularization_params: 
            The smoothing parameters 
        parsed_formula_contents: dict
            A dictionary with keys being parameters of the distribution, e.g. "eta" and "scale"
            and values being dicts with keys deep_models_dict, struct_shapes and P (as used in Sddr_Single)
    Attributes
    ----------
        family: string 
            A string describing the given distribution, e.g. "gaussian", "binomial", ...
        regularization_params: dict
            A dictionary where keys are the name of the distribution parameter (e.g. eta,scale) and values 
            are the smoothing parameters 
        #parameter_names: not used
        
        single_parameter_sddr_list: dict
            A dictionary where keys are the name of the distribution parameter and values are the single_sddr object 
        distribution_layer_type: class object of some type of torch.distributions
            The distribution layer object, defined in the init and depending on the family, e.g. for
            family='normal' the object we will be of type torch.distributions.normal.Normal
        regularization: Torch
            The regularization added to the final loss
        distribution_layer: class instance of some type of torch.distributions
            The final layer of the sddr network, which is initiated depending on the type of distribution (as defined 
            in family) and the predicted parameters from the forward pass
    '''
    
    def __init__(self, family, regularization_params, parsed_formula_contents):
        super(Sddr, self).__init__()
        self.family = family
        self.regularization_params = regularization_params
        #self.parameter_names = parsed_formula_contents.keys
        self.single_parameter_sddr_list = dict()
        for key, value in parsed_formula_contents.items():
            deep_models_dict = value["deep_models_dict"]
            deep_shapes = value["deep_shapes"]
            struct_shapes = value["struct_shapes"]
            P = value["P"]
            self.single_parameter_sddr_list[key] = Sddr_Single(deep_models_dict, deep_shapes, struct_shapes, P)
            
            #register the Sddr_Single network
            self.add_module(key,self.single_parameter_sddr_list[key])
                

        #define distributional layer
        if self.family == "normal":
            self.distribution_layer_type = torch.distributions.normal.Normal
        elif self.family == "poisson":
            self.distribution_layer_type = torch.distributions.poisson.Poisson
    
    def _distribution_trafos(self,pred):
        #applies the specific transformations to the prediction so they they correspond to the restrictions
        #of the parameters
        #this is family specific
        pred_trafo = dict()
        add_const = 1e-8
        
        family = self.family
        if family == "normal":
            pred_trafo["loc"] = pred["loc"]
            pred_trafo["scale"] = add_const + pred["scale"].exp()
        elif family == "poisson":
            pred_trafo["rate"] = add_const + pred["rate"].exp()
        
        return pred_trafo
    
    def forward(self,meta_datadict):
        
        self.regularization = 0
        pred = dict()
        for parameter_name, data_dict  in meta_datadict.items():
            sddr_net = self.single_parameter_sddr_list[parameter_name]
            pred[parameter_name] = sddr_net(data_dict)
            self.regularization += sddr_net.get_regularization()*self.regularization_params[parameter_name]
            
        predicted_parameters = self._distribution_trafos(pred)
        
        #define distributional layer (takes eta and scale)
        self.distribution_layer = self.distribution_layer_type(**predicted_parameters)
        
        return self.distribution_layer
    
    def get_loss(self, Y):
    
#         regularization = 0            # move to forward, or we need meta_datadict as input to get_loss
#         for parameter_name, data_dict  in meta_datadict.items():
#             sddr_net = self.single_parameter_sddr_list[parameter_name]
#             regularization += sddr_net.get_regularization()*self.regularization_params[parameter_name]
        log_loss = -self.distribution_layer.log_prob(Y)
        loss = log_loss + self.regularization
        
        return loss
