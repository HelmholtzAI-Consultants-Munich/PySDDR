import os
import yaml

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# torch imports
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
# pysddr imports
from sddr_network import SddrNet, Sddr_Param_Net
from dataset import SddrDataset
from utils import parse_formulas
from family import Family

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
        degrees_of_freedom: dictionary
            A dictionary where keys are the current distribution's parameters' names and values are the degrees of freedom
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
                                self.config['deep_models_dict'],
                                self.config['train_parameters']['degrees_of_freedom'])

        self.parsed_formula_contents = self.dataset.parsed_formula_content

        self.loader = DataLoader(self.dataset,
                                batch_size=self.config['train_parameters']['batch_size'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device: ', self.device)
        self.net = SddrNet(self.family, self.parsed_formula_contents)
        self.net = self.net.to(self.device)

        # if an optimizer hasn't been defined by the user use adam per default
        if 'optimizer' not in self.config['train_parameters'].keys():
            self.optimizer = optim.Adam(self.net.parameters())
            # save these in the config as we want to save training configuration
            self.config['train_parameters']['optimizer'] = str(self.optimizer)
            self.config['train_parameters']['optimizer_params'] = {'lr': 0.001,
                                                                    'betas': (0.9, 0.999),
                                                                    'eps': 1e-08,
                                                                    'weight_decay': 0,
                                                                    'amsgrad': False}
        else:
            # check if the optimizer is a string not a class instance - will be the case when reading config from yaml
            if isinstance(self.config['train_parameters']['optimizer'],str):
                optimizer = eval(self.config['train_parameters']['optimizer'])
            else:
                optimizer = self.config['train_parameters']['optimizer']
            # if optimizer parameters have been given initialize an optimizer with those
            if 'optimizer_params' in self.config['train_parameters'].keys():
                self.optimizer = optimizer(self.net.parameters(),**self.config['train_parameters']['optimizer_params'])
            # else use torch default parameter values
            else:
                self.optimizer = optimizer(self.net.parameters())
            # and set the optimizer to a string in the config for saving later
            self.config['train_parameters']['optimizer'] = str(self.optimizer)

            
        # check if an output directory has been given - if yes check if it already exists and create it if not
        if self.config['output_dir']:
            if not os.path.exists(self.config['output_dir']):
                os.mkdir(self.config['output_dir'])
    
    def train(self, plot=False):
        '''
        Trains the SddrNet for a number of epochs and prints the loss throughout training
        '''
        self.net.train()
        loss_list = []
        print('Beginning training ...')
        for epoch in range(self.config['train_parameters']['epochs']):
            self.epoch_loss = 0
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
                
                # compute the loss and add regularization
                loss = torch.mean(self.net.get_log_loss(target))
                loss += self.net.get_regularization().squeeze_() 
                
                # and backprobagate
                loss.backward()
                self.optimizer.step()
                self.epoch_loss += loss.item()
                
            # compute the avg loss over all batches in the epoch
            self.epoch_loss = self.epoch_loss/len(self.loader)
            if epoch % 100 == 0:
                print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch, self.epoch_loss))
                
            # and save it in a list in case we want to print later
            loss_list.append(self.epoch_loss)
        if plot:
            plt.plot(loss_list)
            plt.title('Training Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.savefig(os.path.join(self.config['output_dir'], 'train_loss.png'))
            plt.show()
    
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
        # get the weights of the linear layer of the structured part - do this computation on cpu
        structured_head_params = self.net.single_parameter_sddr_list[param].structured_head.weight.detach().cpu()
        # and the structured data after the smoothing
        smoothed_structured = self.dataset.meta_datadict[param]['structured']
        # get a list of the slice that each spline has in the design matrix
        list_of_spline_slices = self.dataset.dm_info_dict[param]['list_of_spline_slices']
        # get a list of the names of spline terms
        list_of_term_names = self.dataset.dm_info_dict[param]['list_of_term_names']
        # get a list of feature names sent as input to each spline
        list_of_spline_input_features = self.dataset.dm_info_dict[param]['list_of_spline_input_features']
        
        partial_effects = []
        can_plot = []
        xlabels = []
        ylabels = []
        
        # for each spline
        for spline_slice, spline_input_features, term_name in zip(list_of_spline_slices, list_of_spline_input_features, list_of_term_names):
            # compute the partial effect = smooth_features * coefs (weights)
            structured_pred = torch.matmul(smoothed_structured[:,spline_slice], structured_head_params[0, spline_slice])
            # if only one feature was sent as input to spline
            if len(spline_input_features) == 1:
                # get that feature
                feature = self.dataset.get_feature(spline_input_features[0])
                # and keep track so that the partial effect of this spline can be plotted later on
                can_plot.append(True)
                ylabels.append(term_name)
                xlabels.append(spline_input_features[0])
            else:
                feature = []
                for feature_name in spline_input_features:
                    feature.append(self.dataset.get_feature(feature_name))
                # the partial effect of this spline cannot be plotted later on - too complicated for now as not 2d
                can_plot.append(False)
            partial_effects.append((feature, structured_pred.numpy()))

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
                    plt.ylabel(ylabels[i])
                    plt.xlabel(xlabels[i])
            plt.tight_layout()
            plt.savefig(os.path.join(self.config['output_dir'], 'partial_effects.png'))
            plt.show()
        return partial_effects
    
    def save(self, name = 'model.pth'):
        """
        Saves the current model's weights and some training conifgurations in a state_dict
        Parameters
        ----------
            name: string
                The name of the file to be saved
        """
        state={
            'epoch': self.config['train_parameters']['epochs'],
            'loss': self.epoch_loss,
            'optimizer': self.optimizer.state_dict(),
            'sddr_net': self.net.state_dict(),
        }
        save_path = os.path.join(self.config['output_dir'], name)
        torch.save(state, save_path)
        train_config_path = os.path.join(self.config['output_dir'], 'train_config.yaml')
        # need to improve
        config = self.config.copy()
        for net in config['deep_models_dict']:
            model = config['deep_models_dict'][net]['model']
            config['deep_models_dict'][net]['model'] = str(model)
        with open(train_config_path, 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

    def load(self):
        """
        Loads a pre-trained model and the used training configuration 
        Parameters
        ----------
            name: string
                The file name from which to load the trained network
        """
        name = self.config['load_model']
        if not torch.cuda.is_available():
            state_dict = torch.load(name, map_location='cpu')
        else:
            state_dict = torch.load(name)
        self.net.load_state_dict(state_dict['sddr_net'])

        if self.config['mode']=='train':
            # Load optimizers
            self.optimizer.load_state_dict(state_dict['optimizer'])
            # Load losses
            self.loss = state_dict['loss']
            epoch = state_dict['epoch']
        print('Loading model {} at epoch {} with a loss {:.4f}'.format(name, epoch, self.loss))

    def inference(self, metadadict):
        """
        Returns the model's prediction
        Parameters
        ----------
            metadadict: dictionary
                A dictionary with the input data structured into structured and unstructured parts
        Returns
        -------
            pred: torch.distributions
                An instance of some torch.distributions class (e.g. torch.distributions.poisson.Poisson) which holds
                also predicted parameters of the distribution(e.g. rate)
        """
        self.net.eval()
        with torch.no_grad():
            pred = self.net(metadadict)
            return pred
    
    def coeff(self, param):
        '''
        Return coefficients (network weights) of the structured head
        '''
        return self.net.single_parameter_sddr_list[param].structured_head.weight.detach().numpy()
    
    def get_distribution(self):
        '''
        Return trained distribution, could be applied .mean/.variance ...
        '''
        return self.net.distribution_layer
    
    '''
    Not yet implemented - we need to write a method which can handle unseen data and make predictions on that
    def predict(self, net_path=None, data=None):
        # not implement yet
        if net_path == None:
            net = self.net
        else:
            net = torch.load(net_path)
            net.eval()
        if data == None:
            data = self.dataset
            
        newdm = build_design_matrices([structured_matrix.design_info], val, return_type='dataframe')[0]  # for prediction (new data) using the same specification of the spline basis.
        
        #distribution_layer = net(data)
        #return distribution_layer
    '''

if __name__ == "__main__":
    params = train()