from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from layers import *
from utils import normalize_adj2, sparse_mx_to_torch_sparse_tensor

# ColaGNN with SIR Constraint in STAN
class ColaGNN_STAN(nn.Module):  
    def __init__(self, args, data): 
        super().__init__()
        self.x_h = 1 
        self.f_h = data.m   
        self.m = data.m  
        self.d = data.d 
        self.w = args.window
        self.h = args.horizon
        # self.adj = data.adj
        # self.o_adj = data.orig_adj
        # if args.cuda:
        #     self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(data.orig_adj.cpu().numpy())).to_dense().cuda()
        # else:
        #     self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(data.orig_adj.cpu().numpy())).to_dense()
        self.adj = torch.eye(self.m)
        if args.cuda:
            self.adj = self.adj.cuda()
        self.dropout = args.dropout
        self.n_hidden = args.n_hidden
        half_hid = int(self.n_hidden/2)
        self.V = Parameter(torch.Tensor(half_hid))
        self.bv = Parameter(torch.Tensor(1))
        self.W1 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.b1 = Parameter(torch.Tensor(half_hid))
        self.W2 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.act = F.elu 
        self.Wb = Parameter(torch.Tensor(self.m,self.m))
        self.wb = Parameter(torch.Tensor(1))
        self.k = args.k
        self.conv = nn.Conv1d(1, self.k, self.w)
        long_kernal = self.w//2
        self.conv_long = nn.Conv1d(self.x_h, self.k, long_kernal, dilation=2)
        long_out = self.w-2*(long_kernal-1)
        self.n_spatial = 10  
        self.conv1 = GraphConvLayer((1+long_out)*self.k, self.n_hidden) # self.k
        self.conv2 = GraphConvLayer(self.n_hidden, self.n_spatial)
 
        if args.rnn_model == 'LSTM':
            self.rnn = nn.LSTM( input_size=self.x_h, hidden_size=self.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        elif args.rnn_model == 'GRU':
            self.rnn = nn.GRU( input_size=self.x_h, hidden_size=self.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        elif args.rnn_model == 'RNN':
            self.rnn = nn.RNN( input_size=self.x_h, hidden_size=self.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        else:
            raise LookupError (' only support LSTM, GRU and RNN')

        hidden_size = (int(args.bi) + 1) * self.n_hidden
        self.out = nn.Linear(hidden_size + self.n_spatial, 1)  

        # SIR parameters
        self.beta_out = nn.Linear(hidden_size + self.n_spatial, 1)  
        self.gamma_out = nn.Linear(hidden_size + self.n_spatial, 1)
        
        self.residual_window = 0
        self.ratio = 1.0
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, args.window)
            self.residual = nn.Linear(self.residual_window, 1) 
            
        self.init_weights()

     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, x, I, R, feat=None):
        '''
        Args:  x: (batch, time_step, m) new cases (normalized)
               I: (batch, time_step, m) total cases
               R: (batch, time_step, m) recovered
            feat: [batch, window, dim, m]
        Returns: (batch, dI)
        ''' 
        b, w, m = x.size()
        orig_x = x 
        x = x.permute(0, 2, 1).contiguous().view(-1, x.size(1), 1) 
        r_out, hc = self.rnn(x, None)
        last_hid = r_out[:,-1,:]
        last_hid = last_hid.view(-1,self.m, self.n_hidden)
        out_temporal = last_hid  # [b, m, 20]
        hid_rpt_m = last_hid.repeat(1,self.m,1).view(b,self.m,self.m,self.n_hidden) # b,m,m,w continuous m
        hid_rpt_w = last_hid.repeat(1,1,self.m).view(b,self.m,self.m,self.n_hidden) # b,m,m,w continuous w one window data
        # a_mx = self.act( hid_rpt_m @ self.W1.t()  + hid_rpt_w @ self.W2.t() + self.b1 ) @ self.V + self.bv # row, all states influence one state 
        # a_mx = F.normalize(a_mx, p=2, dim=1, eps=1e-12, out=None)
        r_l = []
        r_long_l = []
        h_mids = orig_x
        for i in range(self.m):
            h_tmp = h_mids[:,:,i:i+1].permute(0,2,1).contiguous() 
            r = self.conv(h_tmp) # [32, 10/k, 1]
            r_long = self.conv_long(h_tmp)
            r_l.append(r)
            r_long_l.append(r_long)
        r_l = torch.stack(r_l,dim=1)
        r_long_l = torch.stack(r_long_l,dim=1)
        r_l = torch.cat((r_l,r_long_l),-1)
        r_l = r_l.view(r_l.size(0),r_l.size(1),-1)
        r_l = torch.relu(r_l)
        adjs = self.adj.repeat(b,1)
        adjs = adjs.view(b,self.m, self.m)
        # c = torch.sigmoid(a_mx @ self.Wb + self.wb)
        # a_mx = adjs * c + a_mx * (1-c) 
        # adj = a_mx 
        adj = adjs
        x = r_l  
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        out_spatial = F.relu(self.conv2(x, adj))
        final = torch.cat((out_spatial, out_temporal),dim=-1)
        out = self.out(final)
        out = torch.squeeze(out)

        if (self.residual_window > 0):
            z = orig_x[:, -self.residual_window:, :]; #Step backward # [batch, res_window, m]
            z = z.permute(0,2,1).contiguous().view(-1, self.residual_window); #[batch*m, res_window]
            z = self.residual(z); #[batch*m, 1]
            z = z.view(-1,self.m); #[batch, m]
            out = out * self.ratio + z; #[batch, m]
        
        # SIR simulation
        beta = torch.sigmoid(self.beta_out(final))
        self.beta = torch.squeeze(beta)
        gamma = torch.sigmoid(self.beta_out(final))
        self.gamma = torch.squeeze(gamma)
        
        new_I = []
        new_R = []
        for i in range(self.h):
            last_I = I[:, -1, :] if i == 0 else last_I + dI.detach()
            last_R = R[:, -1, :] if i == 0 else last_R + dR.detach()
            
            last_S = 1 - last_I - last_R
            
            dI = self.beta * last_I * last_S - self.gamma * last_I
            dR = self.gamma * last_I

            new_I.append(last_I + dI)
            new_R.append(last_R + dR)
        
        new_I = torch.stack(new_I, dim=1)
        new_R = torch.stack(new_R, dim=1)

        return out, new_I, new_R