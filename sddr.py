import pandas as pd
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from deepregression import SddrNet, Sddr_Param_Net
from dataset import SddrDataset
from utils import parse_formulas, Family
from matplotlib import pyplot as plt

class SDDR(object):
    '''
    The SDDR class is the main class the user interacts with in order to use the PySDDR framework. This class includes
    train, eval functions with which the user can train a pyssddr network and evaluation the training based on the partial
    effects returned.
    Parameters
    ----------
        **kwargs: either a list of parameters or a dict
            The user can give all the necessary parameters either one by one as variables or as a dictionary, where the keys
            are the variables
    Attributes
    -------
        config: dictionary
            A dictionary holding all the user defined parameters
        family: Family 
            An instance of the class Family; on initialization checks whether the distribution given by the user is in the 
            list of available distribution and holds the name of the current distribution defined by the user
        dataset: SddrDataset (inherets torch.utils.data.Dataset)
            An instance that loads the data on initialization, parses the formulas given by the user, splits the data into
            structured and unstructured parts and prepares it for training by the PySSDDR network
        regularization_params: dictionary
            A dictionary where keys are the current distribution's parameters' names and values are the regularization terms
            for each sub-network created for each parameters of the distribution
        parsed_formula_contents: dictionary
            A dictionary where keys are the distribution's parameter names and values are dicts. The keys of these dicts 
            will be: 'struct_shapes', 'P', 'deep_models_dict' and 'deep_shapes' with corresponding values for each distribution
            paramter, i.e. given formula (shapes of structured parts, penalty matrix, a dictionary of the deep models' arcitectures
            used in the current formula and the output shapes of these deep models)
        loader: torch.utils.data.DataLoader
            Used to later load data in batches
        device: torch.device
            The current device
        device: SddrNet 
            The sddr network. This will consist of smaller paraller networks (as many as the parameters of the distribution) each of 
            which is built depending on the formulas and deep models given by the user
        optimizer: torch.optim.RMSprop
            The current optimizer
    '''
    def __init__(self, **kwargs):
        # depending on whether the user has given a dict as input or multiple arguments self.config
        # should be a dict with keys the parameters defined by the user
        for key in kwargs.keys():
            if key == 'config':
                self.config = kwargs['config']
            else:
                self.config = kwargs
            break
        
        # create a family instance
        self.family = Family(self.config['distribution'])
        # create dataset
        self.dataset = SddrDataset(self.config['data'], 
                                self.config['target'],
                                self.family,
                                self.config['formulas'],
                                self.config['deep_models_dict'])

        self.regularization_params = self.config['train_parameters']['regularization_params']

        self.parsed_formula_contents = self.dataset.get_parsed_formula_content()

        self.loader = DataLoader(self.dataset,
                                batch_size=self.config['train_parameters']['batch_size'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = SddrNet(self.family, self.regularization_params, self.parsed_formula_contents)
        self.net = self.net.to(self.device)
        self.optimizer = optim.RMSprop(self.net.parameters())
    
    def train(self):
        '''
        Trains the SddrNet for a number of epochs and prints the loss throughout training
        '''
        self.net.train()
        print('Beginning training ...')
        for epoch in range(self.config['train_parameters']['epochs']):
            for batch in self.loader:
                # for each batch
                target = batch['target'].to(self.device)
                meta_datadict = batch['meta_datadict']          # .to(device) should be improved 
                # send each input batch to the current device
                for param in meta_datadict.keys():
                    for data_part in meta_datadict[param].keys():
                        meta_datadict[param][data_part] = meta_datadict[param][data_part].to(self.device)
                # get the network output
                self.optimizer.zero_grad()
                output = self.net(meta_datadict)
                # compute the loss
                loss = torch.mean(self.net.get_loss(target)) # should target not be sent to device to??
                # and backprobagate
                loss.backward()
                self.optimizer.step()
            if epoch % 100 == 0:
                print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch,loss.item()))
    
    def eval(self, param, plot=True):
        """
        Evaluates the trained SddrNet for a specific parameter of the distribution.

        Parameters
        ----------
            param: string
                The parameter of the distribution for which the evaluation is performed
            plot: boolean, default True
                If true then a figure for each spline defined in the formula of the distribution's parameter is plotted.
                This is only true for the splines which take only one feature as input.
                If false then nothing is plotted.
        Returns
        -------
            partial_effects: list of tuples
                There will be one item in the list for each spline in the distribution's parameter equation. Each item is a tuple
                (feature, partial_effect)
        """
        # get the weights of the linear layer of the structured part
        structured_head_params = self.net.single_parameter_sddr_list[param].structured_head.weight.detach()
        # and the structured data after the smoothing
        smoothed_structured = self.dataset.meta_datadict[param]['structured']
        has_intercept = self.dataset.dm_info_dict[param]['has_intercept']
        # get a list of degrees of freedom for each spline of the distribution's parameter's equaltion
        # number of dofs = number of columns of spline output  
        list_of_dfs = self.dataset.dm_info_dict[param]['list_of_dfs']
        # get a list of feature names sent as input to each spline
        list_of_features = self.dataset.dm_info_dict[param]['list_of_features']
        if has_intercept:
            prev_end = 1
        else:
            prev_end = 0
        partial_effects = []
        can_plot = []
        # for each spline
        for df, feature_names in zip(list_of_dfs, list_of_features):
            # compute the partial effect = smooth_features * coefs (weights)
            structured_pred = torch.matmul(smoothed_structured[:,prev_end:prev_end+df], structured_head_params[0, prev_end:prev_end+df])
            # if only one feature was sent as input to spline
            if len(feature_names) == 1:
                # get that feature
                feature = self.dataset.get_feature(feature_names[0])
                # and keep track so that the partial effect of this spline can be plotted later on
                can_plot.append(True)
            else:
                feature = []
                for feature_name in feature_names:
                    feature.append(self.dataset.get_feature(feature_name))
                # the partial effect of this spline cannot be plotted later on - too complicated for now as not 2d
                can_plot.append(False)
            partial_effects.append((feature, structured_pred.numpy()))
            prev_end = prev_end+df

        if plot:
            num_plots =  sum(can_plot)
            current_non_plots = 0
            if num_plots == 0:
                print('Not possible to print any partial effects')
            elif num_plots != len(partial_effects):
                print('Cannot plot ', len(partial_effects) - num_plots, ' splines because they have more that one input')
            
            plt.figure(figsize=(10,5))
            for i in range(len(partial_effects)):
                if not can_plot[i]:
                    current_non_plots += 1
                else:
                    plt.subplot(1,num_plots,i-current_non_plots+1)
                    feature, partial_effect = partial_effects[i]
                    partial_effect = [x for _,x in sorted(zip(feature, partial_effect))]
                    plt.scatter(np.sort(feature), partial_effect)
                    plt.title('Partial effect %s' % (i+1))
            plt.show()
        return partial_effects
    
    def save(self,name = 'model.pth'):
        torch.save(self.net, name)
    
    def coeff(self, param):
        # return coefficients (network weights) of the structured head
        return self.net.single_parameter_sddr_list[param].structured_head.weight.detach().numpy()
    
    def get_distribution(self):
        # return trained distribution, could be applied .mean/.variance ...
        return self.net.distribution_layer
    
    def predict(self, net_path=None, data=None):
        # not implement yet
        if net_path == None:
            net = self.net
        else:
            net = torch.load(net_path)
            net.eval()
        if data == None:
            data = self.dataset
        
        #distribution_layer = net(data)
        #return distribution_layer
        
    #del load():
    #def inference():

if __name__ == "__main__":
    params = train()