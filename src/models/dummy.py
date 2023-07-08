# Dummy model that always use the last value of the input as the prediction
import torch
import torch.nn as nn

class Dummy(nn.Module): 
    def __init__(self, args, data):
        super().__init__()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = x[:, :, -1]
        return out, None
