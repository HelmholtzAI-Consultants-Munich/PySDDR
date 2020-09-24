import pandas as pd
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from deepregression import SddrNet, Sddr_Param_Net
from dataset import SddrDataset
from utils import parse_formulas, create_family

class SDDR(object):
    def __init__(self, **kwargs):

        for key in kwargs.keys():
            if key == 'config':
                self.config = kwargs['config']
            else:
                self.config = kwargs
            break

        self.family = create_family(self.config['current_distribution'])

        self.dataset = SddrDataset(self.config['data_path'], 
                                self.config['ground_truth_path'],
                                self.family,
                                self.config['formulas'],
                                self.config['deep_models_dict'])

        self.regularization_params = self.config['train_parameters']['regularization_params']

        self.parsed_formula_contents = dataset.get_parsed_formula_content()

        self.loader = DataLoader(dataset,
                                batch_size=self.config['train_parameters']['batch_size'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = SddrNet(self.family_class, self.regularization_params, self.parsed_formula_contents)
        self.net = self.net.to(self.device)
        self.optimizer = optim.RMSprop(self.net.parameters())
        
    def train(self):
        self.net.train()
        print('Begin training ...')
        for epoch in range(self.config['train_parameters']['epochs']):

            for batch in self.loader:
                target = batch['target'].to(self.device)
                meta_datadict = batch['meta_datadict']          # .to(device) should be improved 
                for param in meta_datadict.keys():
                    for key in meta_datadict[param].keys():
                        meta_datadict[param][key] = meta_datadict[param][key].to(self.device)
            
                self.optimizer.zero_grad()
                output = self.net(meta_datadict)
                loss = torch.mean(self.net.get_loss(target))
                loss.backward()
                self.optimizer.step()
            if epoch % 100 == 0:
                print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch,loss.item()))
                
        #return list(bignet.parameters())[0].detach().numpy()

        def eval(self):
            structured_head_params = list(self.net.parameters())[-2].detach().numpy()
            for param in self.self.config['formulas'].keys()
                smoothed_structured = self.dataset.meta_datadict[]

        #del load():
        #def inference():

if __name__ == "__main__":
    params = train()