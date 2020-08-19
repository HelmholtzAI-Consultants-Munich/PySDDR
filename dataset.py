class MyDataset(Dataset):
    def __init__(self):
        x_csv = pd.read_csv (r'./example_data/simple_gam/X.csv',sep=';',header=None)
        y_csv = pd.read_csv (r'./example_data/simple_gam/Y.csv',header=None)
        B_csv = pd.read_csv (r'./example_data/simple_gam/B.csv',sep=';',header=None)
        
        self.struct_data = torch.from_numpy(B_csv.values).float()
        self.deep_data = torch.from_numpy(x_csv.values).float()
        self.y = torch.from_numpy(y_csv.values).float()
        
    def __getitem__(self, index):
        struct = self.struct_data[index]
        deep = self.deep_data[index]
        gt = self.y[index]
        
        datadict = {"structured": struct, "dm1": deep}
        meta_datadict = dict()
        meta_datadict["rate"] = datadict
        
        return {'meta_datadict': meta_datadict, 'target': gt}        
    
    def __len__(self):
        return len(self.deep_data)