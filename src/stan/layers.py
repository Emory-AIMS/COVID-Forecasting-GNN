import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from utils import *

class GraphConvLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        init.xavier_uniform_(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        # feature: N x self.in_feat
        # support: N * self.out_dim
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')' 

 