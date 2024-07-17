import torch
import torch.nn as nn

from UTFF.fft import fft_layer

class TFFblock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TFFblock, self).__init__()

        self.main_layer = nn.Linear(in_channel, out_channel)
        self.fft_layer = fft_layer(in_channel, out_channel)

        self.out_projection = nn.Linear(out_channel, out_channel)

        self.activation = nn.ReLU()

    def forward(self, x):
        main_out = self.main_layer(x)
        fft_out = self.fft_layer(x)
        out = main_out + fft_out

        return self.activation(self.out_projection(out))