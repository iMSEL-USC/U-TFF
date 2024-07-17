import torch
import torch.nn as nn
import torch.fft as fft

class fft_layer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(fft_layer, self).__init__()

        self.real_linear = nn.Linear(in_channel, out_channel)
        self.imag_linear = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        fft_out = fft.fft(x)
        fft_out_real = self.real_linear(fft_out.real)
        fft_out_imag = self.imag_linear(fft_out.imag)
        out = fft.ifft(fft_out_real + fft_out_imag)

        return out.real