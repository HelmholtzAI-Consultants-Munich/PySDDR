import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import parse_formulas

class SddrDataset(Dataset):
    def __init__(self, data, target, family, formulas, deep_models_dict):
        
        if isinstance(data,str):
            self._data = pd.read_csv(data ,sep=None,engine='python')
            self._target = pd.read_csv(target).iloc[:,0].values
        elif isinstance(data,pd.core.frame.DataFrame) and isinstance(target,str):
            self._data = data
            self._target = data[target].values
            
        elif isinstance(data,pd.core.frame.DataFrame) and isinstance(target,pd.core.frame.DataFrame):
            self._data = data
            self._target = target.iloc[:,0].values
            
        self.parsed_formula_content, self.meta_datadict, self.dm_info_dict = parse_formulas(family, formulas, self._data, deep_models_dict)        

        for param in self.meta_datadict.keys():
            for data_part in self.meta_datadict[param].keys():
                self.meta_datadict[param][data_part] = torch.from_numpy(self.meta_datadict[param][data_part]).float()

        self.y = torch.from_numpy(self._target).float()
        
    def get_feature(self, feature_column):
        return self._data.loc[:,feature_column].values
    
    def get_list_of_feature_names(self):
        return list(self._data.columns)
    
        
    def get_parsed_formula_content(self):        
        return self.parsed_formula_content
        
    def __getitem__(self,index):
        batch_data = {}
        for param in self.meta_datadict.keys():
            batch_data[param]={}
            for key in self.meta_datadict[param].keys():
                batch_data[param][key] = self.meta_datadict[param][key][index]
        
        gt = self.y[index]
        return {'meta_datadict': batch_data, 'target': gt}        
    
    def __len__(self):
        return len(self.y)