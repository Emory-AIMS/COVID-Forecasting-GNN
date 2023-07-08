import torch
import torch.nn as nn
from torch.nn import Parameter

class ARMA(nn.Module): 
    def __init__(self, args, data):
        super(ARMA, self).__init__()
        self.m = data.m
        self.w = args.window
        self.n = 2 # larger worse
        self.w = 2*self.w - self.n + 1 
        self.weight = Parameter(torch.Tensor(self.w, self.m)) # 20 * 49
        self.bias = Parameter(torch.zeros(self.m)) # 49
        nn.init.xavier_normal(self.weight)

        args.output_fun = None;
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x):
        x_o = x
        x = x.permute(0,2,1).contiguous()
        n = self.n
        cumsum = torch.cumsum(x,dim=-1)
        cumsum[:,:,n:] = cumsum[:,:,n:] - cumsum[:,:,:-n]
        x = cumsum[:,:,n - 1:] / n
        x = x.permute(0,2,1).contiguous()
        x = torch.cat((x_o,x), dim=1)
        x = torch.sum(x * self.weight, dim=1) + self.bias
        if (self.output != None):
            x = self.output(x)
        return x, None
