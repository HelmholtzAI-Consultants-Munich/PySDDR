import os
import yaml

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# torch imports
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
# pysddr imports
from .sddrnetwork import SddrNet, SddrFormulaNet
from .utils.dataset import SddrDataset
from .utils import checkups
from .utils.prepare_data import PrepareData
from .utils.family import Family
import warnings
import copy

class Sddr(object):
    '''
    The SDDR class is the main class the user interacts with in order to use the PySDDR framework. 
    This class includes 7 functions:
    - 'train' function, with which the user can train a pyssddr network, 
    - 'eval' functions, to evaluate the training based on the partial effects returned, 
    - 'save' and 'load' function, which could save the trained model and load it for furthur usage, 
    - 'coeff' function, to get the coefficients (network weights) of the structured head, 
    - 'get_distribution' function, to get the trained distribution and 
    - 'predict' function, to apply trained model on unseen data.
    
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
        prepare_data: Python Object
            The PrepareData class includes fit and transform functions and parses the formulas defined by the user. 
        device: torch.device
            The current device, e.g. cpu.
        dataset: SddrDataset (inherets torch.utils.data.Dataset)
            An instance that loads the data on initialization, parses the formulas given by the user, splits the data into
            structured and unstructured parts and prepares it for training by the PySSDDR network
        net: SddrNet 
            The sddr network. This will consist of smaller paraller networks (as many as the parameters of the distribution) each 
            of which is built depending on the formulas and deep models given by the user
        loader: torch.utils.data.DataLoader
            Used to later load data in batches
        optimizer: torch optimizer
            The defined torch optimizer, e.g. torch.optim.RMSprop
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
        
        # perform checks on given distribution name, parameter names and number of formulas given
        formulas = checkups(self.family.get_params(), self.config['formulas'])
        self.prepare_data = PrepareData(formulas,
                                        self.config['deep_models_dict'],
                                        self.config['train_parameters']['degrees_of_freedom'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device: ', self.device)

        # check if an output directory has been given - if yes check if it already exists and create it if not
        if 'output_dir' in self.config.keys():
            if not os.path.exists(self.config['output_dir']):
                os.mkdir(self.config['output_dir'])
        else:
            self.config['output_dir'] = './'
    
    def train(self, target, structured_data, unstructured_data=dict(), resume=False, plot=False):
        '''
        Trains the SddrNet for a number of epochs
        
        Parameters
        ----------
            target: str / Pandas.DataFrame 
                target (Y), given as:
                - string: file path pointing to the target column in csv format. This file must contain a column header 
                          that corresponds to the name of the target variable. If target variable (Y) is given as file path, 
                          the input matrix (X) should also be given as file path.
                - string: name of the target variable that will be extracted from the input matrix 'data'. 
                          In this case the taget variable must be contained in the input matrix 'data'.
                - pandas dataframe: the target variable as pandas dataframe column. 
                          In this case the target variable must be excluded from the input matrix 'data'.
            structured_data: str / Pandas.DataFrame
                input dataset (X), given as:
                - string: file path pointing to the input matrix in csv format. This file must contain column headers 
                          that correspond to the names used in the formula. If input matrix (X) is given as file path, 
                          the target variable (Y) should also be given as file path.
                - pandas dataframe: input matrix as pandas object with columns names that correspond to the names used in the    
                          formula.        
            unstructured_data: dictionary - default empty dict
                The information of unstructured data, including file paths of the unstructured data and the type of it (image...)
            resume: bool - default False
                If true, the model could be continue trained based on the loaded results.
            plot: boolean - default False
                If true, the training loss vs epochs could be plotted
        '''
        
        epoch_print_interval = max(1,int(self.config['train_parameters']['epochs']/10))
        
        if resume:
            self.dataset = SddrDataset(structured_data, self.prepare_data, target, unstructured_data, fit=False)
        else:
            self.dataset = SddrDataset(structured_data, self.prepare_data, target, unstructured_data)
            self.net = SddrNet(self.family, self.prepare_data.network_info_dict)
            self.net = self.net.to(self.device)
            self._setup_optim()
            self.cur_epoch = 0

        # get number of train and validation samples
        # e.g. self.config['val_split']= 20 (%)
        n_val = int(len(self.dataset) * self.config['val_split']/100)                                                    
        n_train = len(self.dataset) - n_val
        # split the dataset randomly to train and val
        train, val = random_split(self.dataset, [n_train, n_val])  

        # load train and val data with data loader
        self.train_loader = DataLoader(train, 
                                    batch_size=self.config['train_parameters']['batch_size'])
        self.val_loader = DataLoader(val, 
                                    batch_size=self.config['train_parameters']['batch_size'])
        
        #self.loader = DataLoader(self.dataset,
                                #batch_size=self.config['train_parameters']['batch_size'])

        train_loss_list = []
        val_loss_list = []
        if 'stop_early' in self.config.keys():
            early_stop_counter = 0
            best_loss = 1000000
        
        print('Beginning training ...')
        P = self.prepare_data.get_penalty_matrix(self.device)
        for epoch in range(self.cur_epoch , self.config['train_parameters']['epochs']):
            self.net.train()
            self.epoch_train_loss = 0
            for batch in self.train_loader:
                # for each batch
                target = batch['target'].float().to(self.device)
                datadict = batch['datadict']
                
                # send each input batch to the current device
                for param in datadict.keys():
                    for data_part in datadict[param].keys():
                        datadict[param][data_part] = datadict[param][data_part].float().to(self.device)
                        
                # get the network output
                self.optimizer.zero_grad()
                output = self.net(datadict)
                
                # compute the loss and add regularization
                loss = torch.mean(self.net.get_log_loss(target))
                loss += self.net.get_regularization(P).squeeze_() 
                
                # and backprobagate
                loss.backward()
                self.optimizer.step()
                self.epoch_train_loss += loss.item()
                
            # compute the avg loss over all batches in the epoch
            self.epoch_train_loss = self.epoch_train_loss/len(self.train_loader)
            if epoch % epoch_print_interval == 0:
                print('Train Epoch: {} \t Training Loss: {:.6f}'.format(epoch, self.epoch_train_loss))
            # and save it in a list in case we want to print later
            train_loss_list.append(self.epoch_train_loss)
            
            # after each epoch of training evaluate performance on validation set
            with torch.no_grad():
                self.net.eval()
                epoch_val_loss = 0
                for batch in self.val_loader:
                    # for each batch
                    target = batch['target'].float().to(self.device)
                    datadict = batch['datadict']
                    
                    # send each input batch to the current device
                    for param in datadict.keys():
                        for data_part in datadict[param].keys():
                            datadict[param][data_part] = datadict[param][data_part].float().to(self.device)
                    _ = self.net(datadict)
                    # compute the loss and add regularization
                    val_batch_loss = torch.mean(self.net.get_log_loss(target))
                    val_batch_loss += self.net.get_regularization(P).squeeze_() 
                    epoch_val_loss += val_batch_loss
                epoch_val_loss = epoch_val_loss/len(self.val_loader)
                val_loss_list.append(epoch_val_loss)

                # check if early stopping has been set
                if 'stop_early' in self.config.keys():
                    # criterion can change e.g. get difference and compare to a small valuecd ..
                    if epoch_val_loss < best_loss:
                        best_loss = epoch_val_loss
                        early_stop_counter = 0 
                    else:
                        early_stop_counter += 1

            if epoch % epoch_print_interval == 0:
                print('Train Epoch: {} \t Validation Loss: {:.6f}'.format(epoch, epoch_val_loss))
                    
            if 'stop_early' in self.config.keys() and early_stop_counter == self.config['stop_early']:
                print('Validationloss has not improved for the last %s epochs! To avoid overfitting we are going to stop training now'%(early_stop_counter))
                break

        if plot:
            plt.plot(train_loss_list, label='train')
            plt.plot(val_loss_list, label='validation')
            plt.legend(loc='upper left')
            #plt.title('Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.savefig(os.path.join(self.config['output_dir'], 'train_loss.png'))
            plt.show()
    
    def eval(self, param, plot=True, data=None, get_feature=None):
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
            data: dictionary - default Nonw
                A dictionary where keys are the distribution's parameter names and values are dicts including data in structured 
                and unstructured parts.
            get_feature: numpy array
                The respective feature column from the input matrix
        Returns
        -------
            partial_effects: list of tuples
                There will be one item in the list for each spline in the distribution's parameter equation. Each item is a tuple
                (feature, partial_effect)
        """
        if data == None:
            data = self.dataset[:]["datadict"]
        if get_feature == None:
            get_feature = self.dataset.get_feature
        # get the weights of the linear layer of the structured part - do this computation on cpu
        structured_head_params = self.net.single_parameter_sddr_list[param].structured_head.weight.detach().cpu()
        # and the structured data after the smoothing
        smoothed_structured = data[param]["structured"]
        
        # get a list of the slice that each spline has in the design matrix
        list_of_spline_slices = self.prepare_data.dm_info_dict[param]['spline_info']['list_of_spline_slices']
        # get a list of the names of spline terms
        list_of_term_names = self.prepare_data.dm_info_dict[param]['spline_info']['list_of_term_names']
        
        # get a list of feature names sent as input to each spline
        list_of_spline_input_features = self.prepare_data.dm_info_dict[param]['spline_info']['list_of_spline_input_features']
        
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
                feature = get_feature(spline_input_features[0])
                # and keep track so that the partial effect of this spline can be plotted later on
                can_plot.append(True)
                ylabels.append(term_name)
                xlabels.append(spline_input_features[0])
            else:
                feature = []
                for feature_name in spline_input_features:
                    feature.append(get_feature(feature_name))
                # the partial effect of this spline cannot be plotted later on - too complicated for now as not 2d
                can_plot.append(False)
            partial_effects.append((feature, structured_pred.numpy()))

        if plot:
            num_plots =  sum(can_plot)
            current_non_plots = 0
            if num_plots == 0:
                print('Nothing to plot. No (non-)linear partial effects specified for this parameter. (Deep partial effects are not plotted.)')
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
            'loss': self.epoch_train_loss, # or maybe save validation loss?
            'optimizer': self.optimizer.state_dict(),
            'sddr_net': self.net.state_dict()
        }
        # when it is possible to save this w/o pickle error adapt code
        # 'data_range': self.prepare_data.data_range
        # 'structured_matrix_design_info': self.prepare_data.structured_matrix_design_info,
        warnings.simplefilter('always')
        warnings.warn("""Please note that the metadata for the structured input has not been saved. If you want to load the model and use
        it on new data you will need to also give the structured data used for training as input to the load function.""")
        
        save_path = os.path.join(self.config['output_dir'], name)
        torch.save(state, save_path)
        train_config_path = os.path.join(self.config['output_dir'], 'train_config.yaml')
        # need to improve
        save_config = copy.deepcopy(self.config)
        save_config['train_parameters']['optimizer'] = str(self.optimizer)
        for net in save_config['deep_models_dict']:
            model = save_config['deep_models_dict'][net]['model']
            save_config['deep_models_dict'][net]['model'] = str(model)
        with open(train_config_path, 'w') as outfile:
            yaml.dump(save_config, outfile, default_flow_style=False)
    
    def _load_and_create_design_info(self, training_data, prepare_data):
        # data loader for csv files
        if isinstance(training_data, str):
            training_data = pd.read_csv(training_data, sep=None, engine='python')

        # data loader for Pandas.Dataframe 
        elif isinstance(training_data, pd.core.frame.DataFrame):
            training_data = training_data
        prepare_data.fit(training_data)

    def _setup_optim(self):
        # if an optimizer hasn't been defined by the user use adam per default
        if 'optimizer' not in self.config['train_parameters'].keys():
            self.optimizer = optim.Adam(self.net.parameters())
            # save these in the config as we want to save training configuration
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

    def load(self, model_name, training_data):
        """
        Loads a pre-trained model and the used training configuration 
        Parameters
        ----------
            model_name: string
                The file name from which to load the trained network
            training_data: str / Pandas.DataFrame
                The structured data used to train the model which is loaded here to create the matrix
                design info
        """
        self._load_and_create_design_info(training_data, self.prepare_data)
        if not torch.cuda.is_available():
            state_dict = torch.load(model_name, map_location='cpu')
        else:
            state_dict = torch.load(model_name)

        self.net = SddrNet(self.family, self.prepare_data.network_info_dict)

        #self.prepare_data.set_structured_matrix_design_info(state_dict['structured_matrix_design_info'])
        #self.prepare_data.set_data_range(state_dict['data_range'])
        # move this to predict after model init
        self.net.load_state_dict(state_dict['sddr_net'])
        self.net = self.net.to(self.device)

        self._setup_optim()
        # Load optimizers
        self.optimizer.load_state_dict(state_dict['optimizer'])
        # Load losses
        loss = state_dict['loss']
        self.cur_epoch = state_dict['epoch']
        print('Loaded model {} at epoch {} with a loss {:.4f}'.format(model_name, self.cur_epoch, loss))



    def coeff(self, param):
        '''
        Given a distribution parameter, return the coefficients (network weights) of the corresponding structured head.
        '''
        # get a list of the slices all structured terms inthe data matrix and thus also in the coefficient vector
        list_of_slices = self.prepare_data.dm_info_dict[param]['non_spline_info']['list_of_non_spline_slices']
        list_of_slices += self.prepare_data.dm_info_dict[param]['spline_info']['list_of_spline_slices']


        # get a list of the names all structured terms
        list_of_term_names = self.prepare_data.dm_info_dict[param]['non_spline_info']['list_of_term_names']
        list_of_term_names += self.prepare_data.dm_info_dict[param]['spline_info']['list_of_term_names']

        #get the vector of all coefficients for the structured part for this parameter
        all_coeffs = self.net.single_parameter_sddr_list[param].structured_head.weight.detach().numpy()

        #create dictionary that contains the coefficients for each term in the formula
        coefs_dict = {}
        for term_name, slice_ in zip(list_of_term_names, list_of_slices):
            coefs_dict[term_name] = all_coeffs[0,slice_]

        return coefs_dict   
    
    def get_distribution(self):
        '''
        Return trained distribution, could be applied .mean/.variance ...
        For more choice, please check https://pytorch.org/docs/stable/distributions.html
        '''
        return self.net.distribution_layer
    
    
    def predict(self, data, unstructured_data = False, clipping=False, plot=False):
        """
        Predict and eval on unseen data.
        Parameters
        ----------
            data: Pandas.DataFrame
                The unseen data
            unstructured_data: dictionary - default False
                The information of unstructured data, including file paths of the unstructured data 
                and the data type (e.g. image)
            clipping: boolean, default False
                If true then when the unseen data is out of the range of the training data, they will be clipped.
                If false then when the unseen data is out of range, an error will be thown.
            plot: boolean, default False
                If true, a figure for each spline defined in the formula of the distribution's parameter is plotted.
                This is only true for the splines which take only one feature as input.
        Returns
        -------
            distribution_layer: trained distribution
                The output of the SDDR network, could be applied .mean/.variance ...
            partial_effects: dict of list of tuples
                A dictionary with partial effects for all parameters of the distribution.
                There will be one item in the list for each spline in the distribution's parameter equation. Each item is a tuple
                (feature, partial_effect)
        """
        partial_effects = dict()
        predict_dataset = SddrDataset(data,
                                      prepare_data = self.prepare_data, 
                                      unstructured_data_info = unstructured_data,
                                      fit = False,
                                      clipping = clipping)
        
        datadict = predict_dataset[:]['datadict']
                
        # send each input batch to the current device
        for parameter in datadict.keys():
            for data_part in datadict[parameter].keys():
                datadict[parameter][data_part] = datadict[parameter][data_part].float().to(self.device)
                        
        # get the network output
        with torch.no_grad():
            distribution_layer = self.net(datadict) 
            
        get_feature = lambda feature_name: data.loc[:,feature_name].values
        for param in datadict.keys():
            partial_effects[param] = self.eval(param, plot, data=datadict, get_feature=get_feature)
        
        return distribution_layer, partial_effects

if __name__ == "__main__":
    params = train()