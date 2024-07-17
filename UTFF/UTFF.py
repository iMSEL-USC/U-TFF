import torch
import torch.nn as nn
from UTFF.block import TFFblock
        
class U_TFF(nn.Module):
    def __init__(self, win_size, hidden_dim):
        super(U_TFF, self).__init__()
        
        
        layer_dim = [win_size] + hidden_dim


        self.con1 = TFFblock(layer_dim[0], layer_dim[1])
        self.con2 = TFFblock(layer_dim[1], layer_dim[2])
        self.con3 = TFFblock(layer_dim[2], layer_dim[3])

        self.conT1 = TFFblock(layer_dim[1], layer_dim[0])
        self.conT2 = TFFblock(layer_dim[2], layer_dim[1])
        self.conT3 = TFFblock(layer_dim[3], layer_dim[2])
    
    def forward(self, x):
        out1 = self.con1(x)
        out2 = self.con2(out1)
        out3 = self.con3(out2)

        outtrans3 = self.conT3(out3)
        outtrans2 = self.conT2(outtrans3+out2)
        outtrans1 = self.conT1(outtrans2+out1)
        return outtrans1