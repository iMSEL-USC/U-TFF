import os
import numpy as np

import pandas as pd
import scipy as sp

def RM_read_data(directory, swin):
    ## Input: 
    ##       directory: str, the path to the directory that contains csv dataset file
    ##       swin: int, sliding window size
    ## Output:
    ##       Xdata: (~, swin*number of motors in robotarm), energy consumption segment
    ##       Ydata: (~, 1), label of each segment, normal or abnormal
    
    filenames = os.listdir(directory)
    for fi, filename in enumerate(filenames):
    
        pathname = directory + os.sep + filename
        temp = pd.read_csv(pathname, index_col=(0))
        temp = temp.drop(columns=['time']).to_numpy().astype(np.float32)
        numdata = temp.shape[0]
        
        Xtemp = RM_read_only_energy(temp[:(numdata-swin),2:-10])
        Ytemp = temp[:(numdata-swin),:2]
        for win in range(1, swin):
            Xtemp = np.concatenate((Xtemp,RM_read_only_energy(temp[win:(numdata-(swin-win)),2:-10])), axis=1)
            Ytemp = np.concatenate((Ytemp,temp[win:(numdata-(swin-win)),:2]), axis=1)
        
        if fi == 0:
            Xdata = Xtemp
            Ydata = Ytemp
        else:
            Xdata = np.concatenate((Xdata,Xtemp), axis=0)
            Ydata = np.concatenate((Ydata,Ytemp), axis=0)
    
    return Xdata, Ydata



def RM_read_only_energy(X):
    
    # Remove bad data (some current values are nan, speculated to be 0)
    X[np.isnan(X)] = 0
    
    numdata = X.shape[0]
    X_energy = np.zeros((numdata,5)).astype(np.float32)
    for ndx in range(5):
        
        X_energy[:,ndx] = X[:,1+ndx*4]*X[:,3+ndx*4]*(1e-6)
        
    
    return X_energy