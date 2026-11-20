import numpy as np
from scipy import linalg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
###

def effective_diffusion(cg_data, test_lemonslice, cg_map_grad, M0_red):

    a_loc = test_lemonslice.gamma * np.einsum("kl, kl -> l", cg_map_grad[0,:2], cg_map_grad[0,:2])

    coef, _, _, _ = linalg.lstsq(M0_red,  a_loc) 

    a_loc_nn, x_app, cg_data_tor, x2_scaler, y_scaler, train_losses, val_losses = \
        nn_based(cg_data, a_loc)

    return a_loc, coef, a_loc_nn, x2_scaler, y_scaler

def effective_drift(cg_data, X, test_lemonslice, cg_map_grad, M0_red):
   
    b = test_lemonslice.drift(X)
    b_loc =  np.einsum('jk,jk->k', cg_map_grad[1], b) 

    coef, _, _, _ = linalg.lstsq(M0_red,  b_loc) 

    b_loc_nn, x_app, cg_data_tor, x2_scaler, y_scaler, train_losses, val_losses = \
        nn_based(cg_data, b_loc)

    return b_loc, coef, b_loc_nn, x2_scaler, y_scaler


def nn_based(cg_data, x_loc):
    import torch
    from torch import nn

    cg_data_train, cg_data_val, x_loc_train, x_loc_val = train_test_split(cg_data.T, x_loc[:,None], train_size=0.7, test_size=0.2)
    
    x1_sin_tr = np.sin(cg_data_train[:,0])
    x1_cos_tr = np.cos(cg_data_train[:,0])
    x1_sin_val = np.sin(cg_data_val[:,0])
    x1_cos_val = np.cos(cg_data_val[:,0])

    x2_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    x2_scaled_tr = x2_scaler.fit_transform(cg_data_train[:,1].reshape(-1,1))
    x2_scaled_val = x2_scaler.transform(cg_data_val[:,1].reshape(-1,1))
    
    cg_data_train_processed = np.concatenate((x1_sin_tr[:,None], x1_cos_tr[:,None], x2_scaled_tr),axis=1)
    cg_data_val_processed = np.concatenate((x1_sin_val[:,None], x1_cos_val[:,None], x2_scaled_val),axis=1)

    y_scaler = MinMaxScaler(feature_range=(0, 1))
    x_loc_train_processed = y_scaler.fit_transform(x_loc_train)
    x_loc_val_processed = y_scaler.transform(x_loc_val)

    cg_data_train_processed = torch.tensor(cg_data_train_processed, dtype = torch.float32)
    cg_data_val_processed = torch.tensor(cg_data_val_processed,  dtype = torch.float32)
    x_loc_train_processed = torch.tensor(x_loc_train_processed[:,0], dtype = torch.float32)
    x_loc_val_processed = torch.tensor(x_loc_val_processed[:,0], dtype = torch.float32)

    train_data = torch.utils.data.TensorDataset(cg_data_train_processed, x_loc_train_processed)
    loader_train = torch.utils.data.DataLoader(train_data, batch_size = 200)
   
    ## Define network, loss function and update algorithm
    net = nn.Sequential(nn.Linear(3,8),
                        nn.Tanh(),
                        nn.Linear(8,8),
                        nn.Tanh(),
                        nn.Linear(8,1))
    loss_function = nn.MSELoss()

    lr0 = 0.01
    update = torch.optim.Adam(net.parameters(), lr=lr0, amsgrad=True)  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(update, factor=0.8, patience=10, min_lr=0.0001)
    epochs = 200
    epoch_train_losses = []
    epoch_val_losses = []
    ## Trainings Loop
    for i in range(epochs):
        net.train()
        loss_list = [] 
        for minibatch in loader_train: 
            cg_data_batch, labels = minibatch 

            update.zero_grad()
            output = net(cg_data_batch)[:,0] 

            loss = loss_function(output, labels)
            loss.backward()
            loss_list.append(loss.item())
            update.step()

        epoch_train_losses.append(np.stack(loss_list).mean())

        net.eval()
        x_app = net(cg_data_val_processed)[:,0]

        loss_val = loss_function(x_app,x_loc_val_processed).item()

        scheduler.step(loss_val)

        epoch_val_losses.append(loss_val)

    cg_data_tor = torch.cat((cg_data_train_processed, cg_data_val_processed),0)

    x_app = y_scaler.inverse_transform(net(cg_data_tor).detach())[:,0]

    cg_data_tor = np.concatenate((cg_data_train, cg_data_val),axis=0)
        
    return net, x_app, cg_data_tor, x2_scaler, y_scaler, epoch_train_losses, epoch_val_losses