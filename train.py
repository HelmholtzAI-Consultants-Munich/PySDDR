
## TRAIN and TEST on the small case in example_data/simple_gam
import pandas as pd
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from deepregression import Sddr_Single, Sddr
from dataset import MyDataset
from utils import parse_formulas

def train():
    
    cur_distribution = "poisson"
    family = {'normal':{'loc': 'whateva', 'scale': 'whateva2'}, 'poisson': {'rate': 'whateva'}, 'binomial':{'n': 'whateva', 'p': 'whateva'}}
    
    regularization_params = dict()
    regularization_params["rate"] = 1.   # already mutiplied in full_P
        
    x_path = r'./example_data/simple_gam/X.csv'
    y_path = r'./example_data/simple_gam/Y.csv'
    
    formulas = dict()
    formulas['rate'] = '~1+spline(x1, bs="bs",df=9)+d1(x1)+d2(x2)'
    
    deep_models_dict = dict()
    deep_models_dict['rate'] = dict()
    deep_models_dict['rate']['d1'] = nn.Sequential(nn.Linear(1,10))
    deep_models_dict['rate']['d2'] = nn.Sequential(nn.Linear(1,3),nn.ReLU(), nn.Linear(3,8))
    
    deep_shapes = dict()
    deep_shapes['rate'] = {"d1" : 10, "d2" : 8}
    
    
    dataset = MyDataset(x_path, y_path,family, formulas,cur_distribution, deep_models_dict,deep_shapes)
    loader = DataLoader(
        dataset,
        batch_size=1000,
    )
    parsed_formula_contents = dataset.get_parsed_formula_content()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bignet = Sddr(cur_distribution, regularization_params, parsed_formula_contents)
    bignet = bignet.to(device)
    optimizer = optim.RMSprop(bignet.parameters())

    bignet.train()
    print('Begin training ...')
    for epoch in range(1, 2500):

        for batch in loader:
            target = batch['target'].to(device)
            meta_datadict = batch['meta_datadict']          # .to(device) should be improved 
            meta_datadict['rate']['structured'] = meta_datadict['rate']['structured'].to(device)
            meta_datadict['rate']['d1'] = meta_datadict['rate']['d1'].to(device)
           
            optimizer.zero_grad()
            output = bignet(meta_datadict)
            loss = torch.mean(bignet.get_loss(target))
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch,loss.item()))
            
    return list(bignet.parameters())[0].detach().numpy()

if __name__ == "__main__":
    params = train()