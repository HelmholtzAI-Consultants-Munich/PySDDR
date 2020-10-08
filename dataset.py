import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import parse_formulas

class SddrDataset(Dataset):
    def __init__(self, x_path, y_path,family, formulas, deep_models_dict):
        
        x_csv = pd.read_csv(x_path, names=['x1','x2'],delimiter=';')
        y_csv = pd.read_csv(y_path, header=None)
        
        self.firstCov = x_csv.values[:,0]
        self.parsed_formula_content, self.meta_datadict, self.dm_info_dict = parse_formulas(family, formulas, x_csv, deep_models_dict)        

        for param in self.meta_datadict.keys():
            for data_part in self.meta_datadict[param].keys():
                self.meta_datadict[param][data_part] = torch.from_numpy(self.meta_datadict[param][data_part]).float()

        self.y = torch.from_numpy(y_csv.values).float()
        
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


class SddrDataset_PandasInput(Dataset):
    def __init__(self, data, target_column, family, formulas, deep_models_dict):
        
        
        #self.firstCov = x_csv.values[:,0]
        self.parsed_formula_content, self.meta_datadict, self.dm_info_dict = parse_formulas(family, formulas, data, deep_models_dict)        

        for param in self.meta_datadict.keys():
            for data_part in self.meta_datadict[param].keys():
                self.meta_datadict[param][data_part] = torch.from_numpy(self.meta_datadict[param][data_part]).float()

        self.y = torch.from_numpy(data[target_column]).float()
        
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