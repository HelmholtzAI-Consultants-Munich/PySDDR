import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
import os
#import cv2
import imageio
import numpy as np

class SddrDataset(Dataset):
    '''
    The SDDRDataset class is used to load the data on initialization and parse the formula content of each distribution parameter. 
    The parsing is used to seperate the structured from the unstructured part of the network and to create the corresponding input matrices for each part. It furthermore assembles information about the network structures.
    
    Parameters
    ----------
        data: str / Pandas.DataFrame
            input dataset (X), given as:
            - string: file path pointing to the input matrix in csv format. This file must contain column headers 
                      that correspond to the names used in the formula. If input matrix (X) is given as file path, 
                      the target variable (Y) should also be given as file path.
            - pandas dataframe: input matrix as pandas object with columns names that correspond to the names used in the    
                      formula.
        prepare_data: Python Object
            The Prepare_Data class includes fit and transform functions and parses the formulas defined by the user. 
        target: str / Pandas.DataFrame / None(default)
            target (Y), given as:
            - string: file path pointing to the target column in csv format. This file must contain a column header 
                      that corresponds to the name of the target variable. If target variable (Y) is given as file path, 
                      the input matrix (X) should also be given as file path.
            - string: name of the target variable that will be extracted from the input matrix 'data'. 
                      In this case the taget variable must be contained in the input matrix 'data'.
            - pandas dataframe: the target variable as pandas dataframe column. 
                      In this case the target variable must be excluded from the input matrix 'data'.
            - None (default, normally used in the predict function):
                        If target is given as none a dummy target with zeros will be created. 
        unstructured_data_info: dictionary - default empty dict
            The information of unstructured data, including file paths of the unstructured data and the data type (e.g. image)
        fit: bool - default True
            If the prepare_data object should be fitted to the data during initialization of the dataset.
        clipping: boolean - default False
                If true then when the unseen data is out of the range of the training data, they will be clipped.
                If false then when the unseen data is out of range, an error will be thown.
            
    Attributes
    -------
        unstructured_data_info: dict - default empty dict
            The information of unstructured data, including file paths of the unstructured data and the data type (e.g.image) 
        prepared_data: python object
            The Prepare_Data class includes fit and transform functions and parses the formulas defined by the user. 
        transform: function
            convert to torch object
    '''
    def __init__(self, data, prepare_data, target = None, unstructured_data_info=dict(), fit = True, clipping = False):
        
        try:
            # data loader for csv files
            if isinstance(data,str):
                self._data = pd.read_csv(data ,sep=None,engine='python')
                if isinstance(target, str):
                    self._target = pd.read_csv(target).values
                elif target == None:
                    self._target = np.zeros(len(data))
            
            # data loader for input matrix (X) in Pandas.Dataframe format and target (Y) as feature name (str)
            elif isinstance(data,pd.core.frame.DataFrame):
                self._data = data
                if isinstance(target,str):
                    self._target = data.loc[:,[target]].values
                    self._data = data.drop(target, axis=1)
                elif isinstance(target, pd.core.frame.DataFrame):
                    self._target = target.values
                elif target == None:
                    self._target = np.zeros(len(data))
        except:
            print('ATTENTION! File format for data and target needs to be the same.')

        # add file paths of unstructured features to data
        self.unstructured_data_info = unstructured_data_info

        if self.unstructured_data_info:
            # for testing with local image set uncomment here 
            #self._data = self._data.iloc[:20]
            #self._target = self._target[:20]
            for feature_name in self.unstructured_data_info.keys():
                # if the user hasn't included unstructured data info as column in structured data
                if feature_name not in self._data.columns:
                    list_unstructured_feat_files = os.listdir(self.unstructured_data_info[feature_name]['path'])
                    # remove hidden files
                    list_unstructured_feat_files = [file for file in list_unstructured_feat_files if not file.startswith('.')]
                    # sort them
                    list_unstructured_feat_files.sort()
                    # add this info to data
                    self._data[feature_name] = list_unstructured_feat_files #[:1000]
        if fit:
            prepare_data.fit(self._data)
        self.prepared_data = prepare_data.transform(self._data, clipping) #for the case that there is not so much data it makes sense to preload it here. When we have a lot of batches the transform can also happen in the __getitem__ function.
        self.transform = ToTensor()

    def load_image(self, root_path, image_path):
        img = imageio.imread(os.path.join(root_path, image_path))
        # next 3 lines are used to resize images to size for testing the use of alexnet
        #img = cv2.resize(img,(227,227))
        #img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        #img = Image.open(os.path.join(root_path, image_path))
        img = self.transform(img)
        return img
    
    def load_csv(self, root_path, image_path):
        csv = pd.read_csv(os.path.join(root_path, image_path)).to_numpy()
        # next 3 lines are used to resize images to size for testing the use of alexnet
        #img = cv2.resize(img,(227,227))
        #img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        #img = Image.open(os.path.join(root_path, image_path))
        csv = torch.nn.Flatten(0,1)(self.transform(csv))
        return csv
    
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
                        if cur_feature in self.unstructured_data_info.keys():
                            unstructured_feat_list.append(cur_feature)
                            # for now we can only have one unstructured feature so if it is found break loop
                            break
                    # if there is an unstructured feature - for now only one
                    if unstructured_feat_list:
                        cur_feature = unstructured_feat_list[0]
                        feat_datatype = self.unstructured_data_info[cur_feature]['datatype']
                        root_path = self.unstructured_data_info[cur_feature]['path']
                        if feat_datatype == 'image':
                            if type(index) is int:
                                datadict[param][structured_or_net_name] = self.load_image(root_path, data_row[cur_feature])
                            else:
                                images = []
                                for image_file_name in data_row[cur_feature]:
                                    images.append(self.load_image(root_path, image_file_name).unsqueeze(0))
                                datadict[param][structured_or_net_name] = torch.cat(images)
                                
                        if feat_datatype == 'csv':
                            if type(index) is int:
                                datadict[param][structured_or_net_name] = self.load_csv(root_path, data_row[cur_feature])
                            else:
                                images = []
                                for image_file_name in data_row[cur_feature]:
                                    images.append(self.load_csv(root_path, image_file_name))
                                    
                                ## pad pack sequences:
                                data = torch.nn.utils.rnn.pad_sequence(images, batch_first=True, padding_value=-1.0)
                                
                                data_len = torch.LongTensor(list(map(len, images)))
                                data_packed = torch.nn.utils.rnn.pack_padded_sequence(data, data_len, batch_first=True, enforce_sorted=False)
                                    
                                #datadict[param][structured_or_net_name] = torch.cat(images)
                                print('init data struct')
                                print(images)
                                
                                datadict[param][structured_or_net_name] = data
                                print('padded data')
                                print(data)
                                
                                
                                print('packed data')
                                print(data_packed)
                                
                                print('re-padded data')
                                
                                print(pad_packed_sequence(data_packed, batch_first=True))
                                
                                
                                
                                
                                #datadict[param][structured_or_net_name] = data_packed
                                                      
                                                      
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