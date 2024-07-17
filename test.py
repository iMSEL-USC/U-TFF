import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from loader import RM_read_data
from UTFF.UTFF import U_TFF
from sklearn.metrics import accuracy_score, confusion_matrix


## Define moving average function to smooth error
def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n

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

## change the multi-class label to binary-class label
y_hat = np.minimum(np.maximum(Ytest[:5000,0],0),1)


## Normalization
scaler = MinMaxScaler()
scaler.fit(Xtrain)

Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)


## Initialize data loader for training and testing
train_dataset = torch.from_numpy(Xtrain).float().to(device)
test_dataset = torch.from_numpy(Xtest[:5000,:]).float().to(device)


## Load model and optimizer
hidden_dim = [int(swin*5-3),int(swin*5-6),int(swin*5-9)]
model = U_TFF(swin*5, hidden_dim).to(device)

save_path = "./checkpoint/UTFF_swin"+str(swin)+".pth"
model.load_state_dict(torch.load(save_path)['state_dict'])


## Evaluation

model.eval()

train_out = model(train_dataset)
anomaly_out = model(test_dataset)

train_error = np.average(abs(train_out-train_dataset).detach().cpu().numpy(),axis=1)
anomaly_error = np.average(abs(anomaly_out-test_dataset).detach().cpu().numpy(),axis=1)

ma_train_error = moving_average(train_error,15)
ma_anomaly_error = moving_average(anomaly_error,15)


filename = './result.mat'
sio.savemat(filename, 
            {'train_error':ma_train_error,'error':ma_anomaly_error,'Ytest':y_hat
              })
