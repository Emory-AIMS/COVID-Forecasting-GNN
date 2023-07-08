# -*- coding: utf-8 -*-
import numpy as np
import pickle as pkl
import scipy.sparse as sp
# from scipy.sparse.linalg.eigen.arpack import eigsh
import sys, os
import torch
import re
import string
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
# from scipy.signal import find_peaks
 
 
# define peak area in ground truth data
def peak_error(y_true_states, y_pred_states, threshold): 
    # masked some low values (using training mean by states)
    y_true_states[y_true_states < threshold] = 0
    mask_idx = np.argwhere(y_true_states <= threshold)
    for idx in mask_idx:
        y_pred_states[idx[0]][idx[1]] = 0
    # print(y_pred_states,np.count_nonzero(y_pred_states),np.count_nonzero(y_true_states))
    
    peak_mae_raw = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    peak_mae = np.mean(peak_mae_raw)
    # peak_mae_std = np.std(peak_mae_raw)
    return peak_mae


    
def normalize_adj2(adj):
    """Symmetrically normalize adjacency matrix."""
    # print(adj.shape)
    # adj += sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

 
def normalize(mx):
    """Row-normalize sparse matrix  (normalize feature)"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.float_power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 or len(sparse_mx.col)==0:
        print(sparse_mx.row,sparse_mx.col)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# Smoothing
import numpy as np
from statistics import median

# Smooth county COVID data as in the MATLAB code
def movemedian_6(data):
    data = np.transpose(data)
    smoothed = []
    for node in data:
        l = len(node)
        smoothed_node = []
        for i in range(l):
            lidx, ridx = max(0, i - 3), min(l, i + 3)
            med = median(node[lidx: ridx])
            smoothed_node.append(med)
        smoothed.append(smoothed_node)
    data = np.array(smoothed)
    return np.transpose(data)


# Smooth county COVID data by a 7 day moving mean
def movemean_7(data):
    data = np.transpose(data)
    smoothed = []
    for node in data:
        l = len(node)
        smoothed_node = []
        for i in range(l):
            lidx, ridx = max(0, i - 3), min(l, i + 4)
            mean = np.mean(node[lidx: ridx])
            smoothed_node.append(mean)
        smoothed.append(smoothed_node)
    data = np.array(smoothed)
    return np.transpose(data)
