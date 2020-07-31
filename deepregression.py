#disclaimer: thisis mainly pseudocode


from torch import nn

class Sddr_Single(nn.Module):
    
    def __init__(self, deep_models_dict, struct_shapes, P):
        """
        deep_models_dict: dictionary where key are names of deep models and values are objects that define the deep models
        struct_shapes: number of structural features
        P: numpy matrix for the smoothing regularization (with added zero matrix in the beginning for the linear part)
        
        """
        self.P = P
        self.deep_models_dict = deep_models_dict
        self.structured_head = nn.Linear(struct_shapes,1, bias = False)
        
        output_size_of_deep_models  = get_output_sizes_of_deep_models(deep_models_dict)
        
        self.deep_head = nn.Linear(output_size_of_deep_models,1, bias = False)
              
        
    def _orthog_layer(self, Q, Uhat):
        """
        Utilde = Uhat - QQTUhat
        """
        return Utilde
    
    
    def forward(self, datadict):
        
        X = datadict["structured"]
        Q = torch.qr(X)
        
        Uhat_list = []
        for key in self.deep_models_dict.keys: #assume that the input for the NN has the name of the NN as key
            net = self.deep_models_dict[key]
            Uhat_list.append(net(datadict[key]))
            
        Uhat = torch.cat(Uhat_list) #concatenate the outputs of the deep NNs
        
        Utilde = self._orthog_layer(Q, Uhat)
        
        pred = self.structured_head(X) + self.deep_head(Utilde)
        
        return pred
    
    def get_regularization(self)
        P = torch.from_numpy(self.P) # should have shape struct_shapes x struct_shapes, numpy array
        weights = self.structured_head.weight #should have shape 1 x struct_shapes
        
        
        regularization = torch.matmul(weights.T, torch.matmul(P,weights))
        
        return regularization
        
        
        
class Sddr(nn.Module):
    
    def __init__(self, family, regularization_params, parsed_formula_contents)
        """
        family: string e.g. "gaussian", "binomial"...
        regularization_params: smoothing parameters
        parsed_formula_contents: dictionary with keys being parameters of the distribution, e.g. "eta" and "scale"
        and values being dicts with keys deep_models_dict, struct_shapes and P (as used in Sddr_Single)
        """
        self.family = family
        self.regularization_params = regularization_params
        self.parameter_names = parsed_formula_contents.keys
        self.single_parameter_sddr_list = dict()
        for key, value in parsed_formula_contents:
            deep_models_dict = value["deep_models_dict"]
            struct_shapes = value["struct_shapes"]
            P = value["P"]
            self.single_parameter_sddr_list[key] = Sddr_Single(deep_models_dict, struct_shapes, P))

        #define distributional layer
        if family == "normal":
            self.distribution_layer_type = torch.distributions.normal.Normal
    
    def _distribution_trafos(self,pred):
        #applies the specific transformations to the prediction so they they correspond to the restrictions
        #of the parameters
        #this is family specific
        
        family = self.family
        return predicted_parameters
    
    def forward(self,meta_datadict):
        
        pred = dict()
        for parameter_name, data_dict  in meta_datadict:
            sddr_net = self.single_parameter_sddr_list[parameter_name]
            pred[parameter_name] = sddr_net(data_dict)
            
        predicted_parameters = self._distribution_trafos(pred)
        
        #define distributional layer (takes eta and scale)
        self.distribution_layer = self.distribution_layer_type(**predicted_parameters)
        

        
        return self.distribution_layer
    
    def get_loss(self, Y):
    
        regul = torch.zero()
        for parameter_name, data_dict  in meta_datadict:
            sddr_net = self.single_parameter_sddr_list[parameter_name]
            regul += sddr_net.get_regularization()*self.regularization_params[parameter_name]
        log_loss = self.distributional_layer.log_prob(Y)
        loss = log_loss + regul
        
        return loss

    
    
# missing: train and test
