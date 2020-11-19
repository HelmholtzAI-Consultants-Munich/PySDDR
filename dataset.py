import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
import os
#import cv2
import imageio

class SddrDataset(Dataset):
    '''
    The SDDRDataset class is used to load the data on initialization and parse the formula content of each distribution parameter. 
    The parsing is used to seperate the structured from the unstructured part of the network and to create the corresponding 
    input matrices for each part. It furthermore assembles information about the network structures.
    
    Parameters
    ----------
        data: str / Pandas.DataFrame
            input dataset (X), given as:
            - string: file path pointing to the input matrix in csv format. This file must contain column headers 
                      that correspond to the names used in the formula. If input matrix (X) is given as file path, 
                      the target variable (Y) should also be given as file path.
            - pandas dataframe: input matrix as pandas object with columns names that correspond to the names used in the formula.
        target: str / Pandas.DataFrame
            target (Y), given as:
            - string: file path pointing to the target column in csv format. This file must contain a column header 
                      that corresponds to the name of the target variable. If target variable (Y) is given as file path, 
                      the input matrix (X) should also be given as file path.
            - string: name of the target variable that will be extracted from the input matrix 'data'. 
                      In this case the taget varibel must be contained in the input matrix 'data'.
            - pandas dataframe: the target variable as pandas dataframe column. 
                      In this case the target variable must be excluded from the input matrix 'data'.
        family: Family 
            An instance of the class Family; on initialization checks whether the distribution given by the user is in the 
            list of available distribution and holds the name of the current distribution defined by the user
        formulas: dict
            A dictionary with keys corresponding to the parameters of the distribution defined by the user 
            (e.g. 'rate' for poisson distribution or 'loc' and 'scale' for normal distribution) and values corresponding to strings
            defining the formula for each distribution, e.g. formulas['loc'] = '~ 1 + x1 + spline(x2, bs="bs",df=9) + d1(x1) + d2(x2)'. 
            Formulas must be given in right-sided format only. 
        deep_models_dict: dict
            dictionary where keys are names of the deep models and values are objects that define the deep models
            
        regularization_params: dict
            A dictionary where keys are the name of the distribution parameter (e.g. eta,scale) and values 
            are either a single smooting parameter for all penalities of all splines for this parameter, or a list of smooting parameters, each for one of the splines that appear in the formula for this parameter
            
    Attributes
    -------
        data: Pandas.DataFrame
            input data (X)
        target: Pandas.DataFrame default none
            target (Y)
        y: torch
            target (Y) converted from panda dataframe to torch object
        parsed_formula_contents: dictionary
            A dictionary where keys are the distribution's parameter names and values are dicts. The keys of these dicts 
            will be: 'struct_shapes', 'P', 'deep_models_dict' and 'deep_shapes' with corresponding values for each distribution
            paramter, i.e. given formula (shapes of structured parts, penalty matrix, a dictionary of the deep models' arcitectures
            used in the current formula and the output shapes of these deep models)
        meta_datadict: dictionary
            A dictionary where keys are the distribution's parameter names and values are dicts. The keys of these dicts 
            will be: 'structured' and neural network names if defined in the formula of the parameter (e.g. 'dm1'). Their values 
            are the data for the structured part (after smoothing for the non-linear terms) and unstructured part(s) of the SDDR 
            model 
        dm_info_dict: dictionary
            A dictionary where keys are the distribution's parameter names and values are dicts containing: a bool of whether the
            formula has an intercept or not and a list of the degrees of freedom of the splines in the formula
    '''
    def __init__(self, data, prepare_data, target = None, unstructred_data_info=dict(), fit = True):
        
        try:
            # data loader for csv files
            if isinstance(data,str):
                self._data = pd.read_csv(data ,sep=None,engine='python')
                if isinstance(target, str):
                    self._target = pd.read_csv(target).values
            
            # data loader for input matrix (X) in Pandas.Dataframe format and target (Y) as feature name (str)
            elif isinstance(data,pd.core.frame.DataFrame):
                self._data = data
                if isinstance(target,str):
                    self._target = data.loc[:,[target]].values
                    self._data = data.drop(target, axis=1)
                elif isinstance(target,pd.core.frame.DataFrame):
                    self._target = target.values
        except:
            print('ATTENTION! File format for data and target needs to be the same.')

        # add file paths of unstructured features to data
        self.unstructred_data_info = unstructred_data_info

        if self.unstructred_data_info:
            # for testing with local image set uncomment here 
            #self._data = self._data.iloc[:20]
            #self._target = self._target[:20]
            for feature_name in self.unstructred_data_info.keys():
                # if the user hasn't included unstructured data info as column in structured data
                if feature_name not in self._data.columns:
                    list_unstructured_feat_files = os.listdir(self.unstructred_data_info[feature_name]['path'])
                    # remove hidden files
                    list_unstructured_feat_files = [file for file in list_unstructured_feat_files if not file.startswith('.')]
                    # sort them
                    list_unstructured_feat_files.sort()
                    # add this info to data
                    self._data[feature_name] = list_unstructured_feat_files[:1000]
        if fit:
            prepare_data.fit(self._data)
        self.prepared_data = prepare_data.transform(self._data) #for the case that there is not so much data it makes sense to preload it here. When we have a lot of batches the transform can also happen in the __getitem__ function.
        self.transform = ToTensor()

    def load_image(self, root_path, image_path):
        img = imageio.imread(os.path.join(root_path, image_path))
        # next 3 lines are used to resize images to size for testing the use of alexnet
        #img = cv2.resize(img,(227,227))
        #img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        #img = Image.open(os.path.join(root_path, image_path))
        img = self.transform(img)
        return img

    def __getitem__(self,index):
        datadict = dict()
        found_unstructred = False
        for param in self.prepared_data.keys():
            datadict[param] = dict()
            for structured_or_net_name in self.prepared_data[param].keys():
                unstructured_feat_list = []
                # extract row from pandas data frame
                if type(self.prepared_data[param][structured_or_net_name]) == torch.Tensor:
                    datadict[param][structured_or_net_name] = self.prepared_data[param][structured_or_net_name][index]
                else:
                    data_row = self.prepared_data[param][structured_or_net_name].iloc[index] 
                    if type(index) is int:
                        feature_names = data_row.index
                    else:
                        feature_names = data_row.columns
                    for cur_feature in feature_names:
                        # if there is an unstructured feature it must be read from memory so store that feature in a list
                        if cur_feature in self.unstructred_data_info.keys():
                            unstructured_feat_list.append(cur_feature)
                            # for now we can only have one unstructured feature so if it is found break loop
                            break
                    # if there is an unstructured feature - for now only one
                    if unstructured_feat_list:
                        cur_feature = unstructured_feat_list[0]
                        feat_datatype = self.unstructred_data_info[cur_feature]['datatype']
                        root_path = self.unstructred_data_info[cur_feature]['path']
                        if feat_datatype == 'image':
                            if type(index) is int:
                                datadict[param][structured_or_net_name] = self.load_image(root_path, data_row[cur_feature])
                            else:
                                images = []
                                for image_file_name in data_row[cur_feature]:
                                    images.append(self.load_image(root_path, image_file_name).unsqueeze(0))
                                datadict[param][structured_or_net_name] = torch.cat(images)
                                                      
                            # extend for more cases
                        

        gt = torch.from_numpy(self._target[index]).float()

        return {'datadict': datadict, 'target': gt}        
    
    def __len__(self):
        return len(self._target)
    
    def get_feature(self, feature_column):
        """
        For a given feature name, extract the respective column from the input matrix (data - without target columns)
        """
        return self._data.loc[:,feature_column].values
    
    def get_list_of_feature_names(self):
        """
        Get the names of all input features (column names of input matrix (data - without target columns)).
        """
        return list(self._data.columns)