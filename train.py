
## TRAIN and TEST on the small case in example_data/simple_gam

def train():
    
    family = "poisson"
    
    regularization_params = dict()
    regularization_params["rate"] = 1.   # already mutiplied in full_P
    
    deep_models_dict = {}
    deep_shapes = {}
    struct_shapes = 19
    P = pd.read_csv (r'./example_data/simple_gam/full_P.csv',sep=';',header=None).values
    
    parsed_formula_contents = dict()
    parsed_formula_contents["rate"] = {"deep_models_dict": deep_models_dict, "deep_shapes": deep_shapes, "struct_shapes": struct_shapes, "P": P}
    

    dataset = MyDataset()
    loader = DataLoader(
        dataset,
        batch_size=1000,
    )
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bignet = Sddr(family, regularization_params, parsed_formula_contents)
    bignet = bignet.to(device)
    optimizer = optim.RMSprop(bignet.parameters())

    bignet.train()
    print('Begin training ...')
    for epoch in range(1, 2500):

        for batch in loader:
            target = batch['target'].to(device)
            meta_datadict = batch['meta_datadict']          # .to(device) should be improved 
            meta_datadict['rate']['structured'] = meta_datadict['rate']['structured'].to(device)
            meta_datadict['rate']['dm1'] = meta_datadict['rate']['dm1'].to(device)
           
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