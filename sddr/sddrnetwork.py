import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd


## SDDR NETWORK PART
class SddrFormulaNet(nn.Module):
    '''
    This class represents an sddr network with a structured part, one or many deep models, a linear layer for 
    the structured part and a linear layer for the concatenated outputs of the deep models. The concatenated 
    outputs of the deep models are first filtered with an orthogonilization layer which removes any linear 
    parts of the deep output (by taking the Q matrix of the QR decomposition of the output of the structured part). 
    The two outputs of the linear layers are added so a prediction of a single parameter of the distribution is made
    and is returned as the final output of the network.
    The model follows the architecture described in Read me file.

    Parameters
    ----------
    deep_models_dict: dict
        dictionary where keys are names of the deep models and values are objects that define the deep models
    deep_shapes: dict
        dictionary where keys are names of the deep models and values are the outputs shapes of the deep models
    struct_shapes: int
        number of structural features
    orthogonalization_pattern: list of slice objects
        orthogonalization patterns for the deep neural networks, For each term in the design matrix wrt that the deep neural 
        network should be orthogonalized there is a slice in the list.
    p: float
        Dropout rate, probability of an element to be zeroed, the recommend value should between 0.01-0.1(depending on the feature numbers). The dropout is used for eastimate uncertainty.
        
    Attributes
    ----------
    deep_models_dict: dict
        dictionary where keys are names of the deep models and values are objects that define the deep models
     orthogonalization_pattern: list of slice objects
        orthogonalization patterns for the deep neural networks
    structured_head: nn.Linear
        A linear layer which is fed the structured part of the data
    deep_head: nn.Linear
        A linear layer which is fed the unstructured part of the data
    deep_models_exist: Boolean
        This value is true if deep models have been used on init of the ssdr_single network, otherwise it is false
    '''
    
    def __init__(self, deep_models_dict, deep_shapes, struct_shapes, orthogonalization_pattern, p):
        
        super(SddrFormulaNet, self).__init__()
        self.deep_models_dict = deep_models_dict
        
        #register external neural networks
        for key, value in deep_models_dict.items():
            self.add_module(key,value)
        
        self.orthogonalization_pattern = orthogonalization_pattern
        if struct_shapes == 0:
            self.structured_head = Zero_Layer()
        else:
            self.structured_head = nn.Linear(struct_shapes,1, bias = False)
        
        if len(deep_models_dict) != 0:
            output_size_of_deep_models  = sum([deep_shapes[key] for key in deep_shapes.keys()])
            self.deep_head = nn.Linear(output_size_of_deep_models,1, bias = False)
            self._deep_models_exist = True
        else:
            self._deep_models_exist = False
        
        self.p = p     
        
    def _orthog_layer(self, Q, Uhat):
        """
        Utilde = Uhat - QQTUhat
        """
        
        Projection_Matrix = Q @ Q.T
        Utilde = Uhat - Projection_Matrix @ Uhat
        
        return Utilde
    
    
    def forward(self, datadict,training=True):
        X = datadict["structured"]
        print(datadict["structured"])
        print(datadict["structured"].shape)
        
        if self._deep_models_exist:

            Utilde_list = []
            for key in self.deep_models_dict.keys(): #assume that the input for the NN has the name of the NN as key
                net = self.deep_models_dict[key]
                
                Uhat_net = net(datadict[key])
                print(key)
                #print(datadict[key])
                
                # orthogonalize the output of the neural network with respect to the parts of the structured part,
                # that contain the same input as the neural network
                #if len(self.orthogonalization_pattern[key]) >0:
                X_sliced_with_orthogonalization_pattern = X # torch.cat([X[:,sl] for sl in self.orthogonalization_pattern[key]],1)
                Q, R = torch.qr(X_sliced_with_orthogonalization_pattern)
                Utilde_net = self._orthog_layer(Q, Uhat_net)
                    #print(Q)
               # else:
                    #print('self.orthogonalization_pattern[key]')
                    #print(self.orthogonalization_pattern[key])
                #    Utilde_net = Uhat_net
                
                Utilde_list.append(Utilde_net)
            
            Utilde = torch.cat(Utilde_list, dim = 1) #concatenate the orthogonalized outputs of the deep NNs
            
            Utilde = nn.functional.dropout(Utilde,p=self.p,training=training)            
            deep_pred = self.deep_head(Utilde)
        else:
            deep_pred = 0
                
        X = nn.functional.dropout(X,p=self.p,training=training)        
        structured_pred = self.structured_head(X)
        
        pred = structured_pred + deep_pred
        

        return pred
    
    def get_regularization(self, P):
        '''
        P = torch.from_numpy(P).float() # should have shape struct_shapes x struct_shapes, numpy array
        # do this somewhere else in the future?
        P = P.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        '''
        try:
            weights = self.structured_head.weight #should have shape 1 x struct_shapes
            regularization = weights @ P @ weights.T
        except:
            regularization = 0
        return regularization
        
        
class SddrNet(nn.Module):
    '''
    This class represents the full sddr network which can consist of one or many smaller sddr nets (in a parallel manner).
    Each smaller sddr predicts one distribution parameter and these are then sent into a transformation layer which applies
    constraints on the parameters depending on the given distribution. The output parameters are then fed into a distributional
    layer and a log-loss is computed. A regularization term is added to the log-loss to compute the total loss of the network.
    The model follows the architecture depicted here:
    https://docs.google.com/presentation/d/1cBgh9LoMNAvOXo2N5t6xEp9dfETrWUvtXsoSBkgVrG4/edit#slide=id.g8ed34c120e_5_16

    Parameters
    ----------
        family: Family 
            An instance of the class Family
        network_info_dict: dict
            A dictionary with keys being parameters of the distribution, e.g. "eta" and "scale"
            and values being dicts with keys deep_models_dict, deep_shapes, struct_shapes and orthogonalization_pattern
        p: float
            Dropout rate, probability of an element to be zeroed, the recommend value should between 0.01-0.1(depending on the feature numbers). The dropout is used for uncertainty-estimation.
            
    Attributes
    ----------
        family: Family 
            An instance of the class Family        
        single_parameter_sddr_list: dict
            A dictionary where keys are the name of the distribution parameter and values are the single_sddr object 
        distribution_layer_type: class object of some type of torch.distributions
            The distribution layer object, defined in the init and depending on the family, e.g. for
            family='normal' the object we will be of type torch.distributions.normal.Normal
        distribution_layer: class instance of some type of torch.distributions
            The final layer of the sddr network, which is initiated depending on the type of distribution (as defined 
            in family) and the predicted parameters from the forward pass
    '''
    
    def __init__(self, family, network_info_dict, p):
        super(SddrNet, self).__init__()
        self.family = family
        self.single_parameter_sddr_list = dict()
        for key, value in network_info_dict.items():
            deep_models_dict = value["deep_models_dict"]
            deep_shapes = value["deep_shapes"]
            struct_shapes = value["struct_shapes"]
            orthogonalization_pattern = value["orthogonalization_pattern"]
            self.single_parameter_sddr_list[key] = SddrFormulaNet(deep_models_dict, 
                                                                  deep_shapes, 
                                                                  struct_shapes, 
                                                                  orthogonalization_pattern,
                                                                  p)
            
            #register the SddrFormulaNet network
            self.add_module(key,self.single_parameter_sddr_list[key])
                
        self.distribution_layer_type = family.get_distribution_layer_type()
        
    def forward(self,datadict,training=True):
        
        self.regularization = 0
        pred = dict()
        for parameter_name, data_dict_param  in datadict.items():
            sddr_net = self.single_parameter_sddr_list[parameter_name]
            pred[parameter_name] = sddr_net(data_dict_param,training=training)
            
        predicted_parameters = self.family.get_distribution_trafos(pred)
        
        self.distribution_layer = self.distribution_layer_type(**predicted_parameters)
        
        return self.distribution_layer
    
    def get_log_loss(self, Y):
        ''' Compute log loss based on the trained distributional layer and the groundtruth Y '''
        log_loss = -self.distribution_layer.log_prob(Y)
        
        return log_loss
    
    def get_regularization(self, P):
        ''' Compute regularization given penalty matrix P '''
        regularization = 0
        for param  in self.single_parameter_sddr_list.keys():
            sddr_net = self.single_parameter_sddr_list[param]
            regularization += sddr_net.get_regularization(P[param])
        return regularization

    
class Zero_Layer(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Zero_Layer, self).__init__()
        self.weight = torch.tensor(0)

    def forward(self, input):
        return torch.tensor(0)
