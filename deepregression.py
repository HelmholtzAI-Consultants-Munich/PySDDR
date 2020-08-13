import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd


## SDDR NETWORK PART
class Sddr_Single(nn.Module):
    
    def __init__(self, deep_models_dict, deep_shapes, struct_shapes, P):
        """
        deep_models_dict: dictionary where key are names of deep models and values are objects that define the deep models
        struct_shapes: number of structural features
        P: numpy matrix for the smoothing regularization (with added zero matrix in the beginning for the linear part)
        
        """
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
    
    def __init__(self, family, regularization_params, parsed_formula_contents):
        """
        family: string e.g. "gaussian", "binomial"...
        regularization_params: smoothing parameters
        parsed_formula_contents: dictionary with keys being parameters of the distribution, e.g. "eta" and "scale"
        and values being dicts with keys deep_models_dict, struct_shapes and P (as used in Sddr_Single)
        """
        super(Sddr, self).__init__()
        self.family = family
        self.regularization_params = regularization_params
        self.parameter_names = parsed_formula_contents.keys
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
        if family == "normal":
            self.distribution_layer_type = torch.distributions.normal.Normal
        elif family == "poisson":
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
        
        self.regul = 0
        pred = dict()
        for parameter_name, data_dict  in meta_datadict.items():
            sddr_net = self.single_parameter_sddr_list[parameter_name]
            pred[parameter_name] = sddr_net(data_dict)
            self.regul += sddr_net.get_regularization()*self.regularization_params[parameter_name]
            
        predicted_parameters = self._distribution_trafos(pred)
        
        #define distributional layer (takes eta and scale)
        self.distribution_layer = self.distribution_layer_type(**predicted_parameters)
        
        return self.distribution_layer
    
    def get_loss(self, Y):
    
#         regul = 0            # move to forward, or we need meta_datadict as input to get_loss
#         for parameter_name, data_dict  in meta_datadict.items():
#             sddr_net = self.single_parameter_sddr_list[parameter_name]
#             regul += sddr_net.get_regularization()*self.regularization_params[parameter_name]
        log_loss = -self.distribution_layer.log_prob(Y)
        loss = log_loss + self.regul
        
        return loss
        
        


        
        
## TRAIN and TEST on the small case in example_data/simple_gam/

class MyDataset(Dataset):
    def __init__(self):
        x_csv = pd.read_csv (r'./example_data/simple_gam/X.csv',sep=';',header=None)
        y_csv = pd.read_csv (r'./example_data/simple_gam/Y.csv',header=None)
        B_csv = pd.read_csv (r'./example_data/simple_gam/B.csv',sep=';',header=None)
        
        self.struct_data = torch.from_numpy(B_csv.values).float()
        self.deep_data = torch.from_numpy(x_csv.values).float()
        self.y = torch.from_numpy(y_csv.values).float()
        
    def __getitem__(self, index):
        struct = self.struct_data[index]
        deep = self.deep_data[index]
        gt = self.y[index]
        
        datadict = {"structured": struct, "dm1": deep}
        meta_datadict = dict()
        meta_datadict["rate"] = datadict
        
        return {'meta_datadict': meta_datadict, 'target': gt}        
    
    def __len__(self):
        return len(self.deep_data)


def train():
    
    family = "poisson"
    
    regularization_params = dict()
    regularization_params["rate"] = 1.   # already mutiplied in full_P
    
    deep_models_dict = {}
    deep_shapes = {}
    struct_shapes = 19
    P = pd.read_csv (r'./example_data/simple_gam/full_P.csv',sep=';',header=None).values
    
    parsed_formula_contents = dict()
    parsed_formula_contents["rate"] = {"deep_models_dict": deep_models_dict, "deep_shapes": deep_shapes, "struct_shapes": struct_shapes, "P": P}
    

    dataset = MyDataset()
    loader = DataLoader(
        dataset,
        batch_size=1000,
    )
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bignet = Sddr(family, regularization_params, parsed_formula_contents)
    bignet = bignet.to(device)
    optimizer = optim.RMSprop(bignet.parameters())

    bignet.train()
    print('Begin training ...')
    for epoch in range(1, 2500):

        for batch in loader:
            target = batch['target'].to(device)
            meta_datadict = batch['meta_datadict']          # .to(device) should be improved 
            meta_datadict['rate']['structured'] = meta_datadict['rate']['structured'].to(device)
            meta_datadict['rate']['dm1'] = meta_datadict['rate']['dm1'].to(device)
           
            optimizer.zero_grad()
            output = bignet(meta_datadict)
            loss = torch.mean(bignet.get_loss(target))
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch,loss.item()))
            
    return list(bignet.parameters())[0].detach().numpy()
