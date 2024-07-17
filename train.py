import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from loader import RM_read_data
from UTFF.UTFF import U_TFF



device = 'cuda'

## Load data
path = "./data"
swin = 3

directory = path + '/Data1'
Xtrain, Ytrain = RM_read_data(directory, swin)
directory = path + '/Data2'
Xtest, Ytest = RM_read_data(directory, swin)

Ytrain = Ytrain[:,-2:]
Ytest = Ytest[:,-2:]


## Normalization
scaler = MinMaxScaler()
scaler.fit(Xtrain)

Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)


## Initialize data loader for training and testing
batch_size = 1024
train_dataset = torch.from_numpy(Xtrain).float().to(device)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_dataset = torch.from_numpy(Xtest).float().to(device)


## Initialize model and optimizer
hidden_dim = [int(swin*5-3),int(swin*5-6),int(swin*5-9)]
model = U_TFF(swin*5, hidden_dim).to(device)

lr = 1e-3
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


## Path to save checkpoint
save_path = "./checkpoint/UTFF_swin"+str(swin)+".pth"


## Training process
max_val = np.inf
max_epoch = 1000
for epoch in range(max_epoch):
    model.train()
    epoch_loss = 0

    for _, data in enumerate(train_loader):
        data = data.float().to(device)
        optimizer.zero_grad()
        out = model(data)

        loss = loss_fn(out, data)
        loss.backward()
        optimizer.step()

        epoch_loss += loss

    model.eval()
    val_out = model(train_dataset)    
    val_loss = loss_fn(val_out,train_dataset).detach().cpu().numpy()

    print("Epoch {}: Val Loss:{:.8f}".format(epoch, val_loss))
    epoch += 1
    if val_loss <= max_val:
        max_val = val_loss
        checkpoint = {'model':model,
                      'state_dict':model.state_dict(),
                      'optimizer':optimizer.state_dict()}
        torch.save(checkpoint, save_path)