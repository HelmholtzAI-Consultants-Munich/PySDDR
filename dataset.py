import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import parse_formulas

'''
x_path = r'./example_data/simple_gam/X.csv'
y_path = r'./example_data/simple_gam/Y.csv'
b_path = r'./example_data/simple_gam/B.csv'
'''
class SddrDataset(Dataset):
    def __init__(self, x_path, y_path,family, formulas,cur_distribution, deep_models_dict,deep_shapes):
        
        x_csv = pd.read_csv(x_path, names=['x1','x2'],delimiter=';')
        y_csv = pd.read_csv(y_path, header=None)
        
        self.parsed_formula_content, self.meta_datadict = parse_formulas(family, formulas, x_csv, cur_distribution, deep_models_dict, deep_shapes)        
  
        for param in self.meta_datadict.keys():
            for key in self.meta_datadict[param].keys():
                self.meta_datadict[param][key] = torch.from_numpy(self.meta_datadict[param][key]).float()

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
