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
    def __init__(self, **kwargs):

        for key in kwargs.keys():
            if key == 'config':
                self.config = kwargs['config']
            else:
                self.config = kwargs
            break

        self.family = Family(self.config['current_distribution'])

        self.dataset = SddrDataset(self.config['data_path'], 
                                self.config['ground_truth_path'],
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
        self.net.train()
        print('Begin training ...')
        for epoch in range(self.config['train_parameters']['epochs']):

            for batch in self.loader:
                target = batch['target'].to(self.device)
                meta_datadict = batch['meta_datadict']          # .to(device) should be improved 
                for param in meta_datadict.keys():
                    for data_part in meta_datadict[param].keys():
                        meta_datadict[param][data_part] = meta_datadict[param][data_part].to(self.device)
            
                self.optimizer.zero_grad()
                output = self.net(meta_datadict)
                loss = torch.mean(self.net.get_loss(target))
                loss.backward()
                self.optimizer.step()
            if epoch % 100 == 0:
                print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch,loss.item()))
        
        #return list(bignet.parameters())[0].detach().numpy()
    
    def eval(self, param, plot=True):

        structured_head_params = self.net.single_parameter_sddr_list[param].structured_head.weight.detach()
        smoothed_structured = self.dataset.meta_datadict[param]['structured']
        has_intercept, list_of_dfs = self.dataset.dm_info_dict[param]['has_intercept'], self.dataset.dm_info_dict[param]['list_of_dfs']
        if has_intercept:
            prev_end = 1
        else:
            prev_end = 0
        partial_effects = []
        for df in list_of_dfs:
            hatY_pred = torch.matmul(smoothed_structured[:,prev_end:prev_end+df], structured_head_params[0, prev_end:prev_end+df])
            partial_effect = [x for _,x in sorted(zip(self.dataset.firstCov, hatY_pred.numpy()))]
            partial_effects.append((self.dataset.firstCov, partial_effect))
            prev_end = prev_end+df
        if plot:
            num_plots = len(partial_effects)
            plt.figure(figsize=(10,5))
            for i in range(num_plots):
                plt.subplot(1,num_plots,i+1)
                firstCov, partial_effect = partial_effects[i]
                plt.scatter(np.sort(firstCov), partial_effect)
                plt.title('Partial effect %s' % (i))
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