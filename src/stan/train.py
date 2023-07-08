from data_downloader import GenerateTrainingData
import pickle
import pandas as pd
import numpy as np
import argparse
import torch

import os, random, argparse, time
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,explained_variance_score
from math import sqrt

import scipy.sparse as sp
# from scipy.stats.stats import pearsonr
from scipy.stats import pearsonr
from colagnn_stan import *
from data import *

import shutil
import logging
import glob
import time
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp
import warnings
warnings.filterwarnings("ignore")

FIPS = [6001, 6005, 6007, 6009, 6013, 6015, 6017, 6019, 6021, 6023, 6025, 6029, 6031, 6033, 6037, 6039, 
        6041, 6045, 6047, 6053, 6055, 6057, 6059, 6061, 6065, 6067, 6069, 6071, 6073, 6075, 6077, 6079, 
        6081, 6083, 6085, 6087, 6089, 6093, 6095, 6097, 6099, 6101, 6103, 6107, 6109, 6111, 6113, 6115]
COUNTY = ['Alameda', 'Amador', 'Butte', 'Calaveras', 'Contra Costa', 'Del Norte', 'El Dorado', 'Fresno', 
          'Glenn', 'Humboldt', 'Imperial', 'Kern', 'Kings', 'Lake', 'Los Angeles', 'Madera', 'Marin', 
          'Mendocino', 'Merced', 'Monterey', 'Napa', 'Nevada', 'Orange', 'Placer', 'Riverside', 'Sacramento', 
          'San Benito', 'San Bernardino', 'San Diego', 'San Francisco', 'San Joaquin', 'San Luis Obispo', 
          'San Mateo', 'Santa Barbara', 'Santa Clara', 'Santa Cruz', 'Shasta', 'Siskiyou', 'Solano', 
          'Sonoma', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Tuolumne', 'Ventura', 'Yolo', 'Yuba']

# # Generate Data, saved to './data/state_covid_data.pickle'
# GenerateTrainingData().download_jhu_data('2020-04-06', '2021-10-06')

# Load time series data and population data
raw_data = pickle.load(open('./data/state_covid_data.pickle','rb'))
raw_data = raw_data[raw_data['fips'].isin(FIPS)]
pop_data = pd.read_csv('./data/uszips.csv')
pop_data = pop_data.rename(columns={"county_fips": 'fips'})
pop_data = pop_data.groupby('fips').agg({'population':'sum'}).reset_index()
pop_data = pop_data[pop_data['fips'].isin(FIPS)]
pop_data = pop_data[['fips', 'population']]

# Preprocess features
active_cases = []
confirmed_cases = []
death_cases = []
county_pop = []

for fips in FIPS:
        active_cases.append(raw_data[raw_data['fips'] == fips]['active'])
        confirmed_cases.append(raw_data[raw_data['fips'] == fips]['confirmed'])
        death_cases.append(raw_data[raw_data['fips'] == fips]['deaths'])
        county_pop.append(pop_data[pop_data['fips'] == fips]['population'])

active_cases = np.array(active_cases)
confirmed_cases = np.array(confirmed_cases)
death_cases = np.array(death_cases)
county_pop = np.array(county_pop)
recovered_cases = confirmed_cases - active_cases - death_cases
susceptible_cases = np.repeat(county_pop, active_cases.shape[1], -1) - active_cases - recovered_cases

# Batch_feat: new_cases(dI), dR, dS
dI = np.transpose(np.diff(confirmed_cases))
dR = np.transpose(np.diff(recovered_cases))
dS = np.transpose(np.diff(susceptible_cases))

np.savetxt("ts.txt", dI, fmt='%.1f', delimiter=',')

active_cases = np.transpose(active_cases[:, 1:])
confirmed_cases = np.transpose(confirmed_cases[:, 1:])
death_cases = np.transpose(death_cases[:, 1:])
county_pop = np.transpose(county_pop)
recovered_cases = np.transpose(recovered_cases[:, 1:])
susceptible_cases = np.transpose(susceptible_cases[:, 1:])

infected = active_cases / county_pop
recovered = recovered_cases / county_pop

# Training settings
ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, default='ca48-548', help="Dataset string")
ap.add_argument('--sim_mat', type=str, default='ca48-adj', help="adjacency matrix filename (*-adj.txt)")
ap.add_argument('--sci', type=str, default='ca48-sci', help="social connectednes index")
ap.add_argument('--svi', type=str, default='', help="social vulnerability index data")
ap.add_argument('--n_layer', type=int, default=1, help="number of layers (default 1)") 
ap.add_argument('--n_hidden', type=int, default=20, help="rnn hidden states (could be set as any value)") 
ap.add_argument('--seed', type=int, default=42, help='random seed')
ap.add_argument('--epochs', type=int, default=1500, help='number of epochs to train')
ap.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
ap.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
ap.add_argument('--dropout', type=float, default=0.2, help='dropout rate usually 0.2-0.5.')
ap.add_argument('--batch', type=int, default=32, help="batch size")
ap.add_argument('--check_point', type=int, default=1, help="check point")
ap.add_argument('--shuffle', action='store_true', default=False, help="not used, default false")
ap.add_argument('--train', type=float, default=.7, help="Training ratio (0, 1)")
ap.add_argument('--val', type=float, default=.15, help="Validation ratio (0, 1)")
ap.add_argument('--test', type=float, default=.15, help="Testing ratio (0, 1)")
ap.add_argument('--model', default='colagnn_stan', help='Model to use')
ap.add_argument('--rnn_model', default='RNN', choices=['LSTM','RNN','GRU'], help='')
ap.add_argument('--mylog', action='store_false', default=True,  help='save tensorboad log')
ap.add_argument('--cuda', action='store_true', default=True,  help='')
ap.add_argument('--window', type=int, default=28, help='') 
ap.add_argument('--horizon', type=int, default=5, help='leadtime default 1') 
ap.add_argument('--save_dir', type=str,  default='save',help='dir path to save the final model')
ap.add_argument('--gpu', type=int, default=1,  help='choose gpu 0-10')
ap.add_argument('--lamda', type=float, default=0.01,  help='regularize params similarities of states')
ap.add_argument('--bi', action='store_true', default=False,  help='bidirectional default false')
ap.add_argument('--patience', type=int, default=100, help='patience default 100')
ap.add_argument('--k', type=int, default=10,  help='kernels')
ap.add_argument('--hidsp', type=int, default=15,  help='spatial dim')
ap.add_argument('--smoothf', type=str, default="movemean_7", choices=['movemean_7', 'movemedian_6', 'none'], help='util function used to smooth the input time series data')

args = ap.parse_args() 
print('--------Parameters--------')
print(args)
print('--------------------------')

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.cuda = args.cuda and torch.cuda.is_available() 
logger.info('cuda %s', args.cuda)

time_token = str(time.time()).split('.')[0] # tensorboard model
log_token = '%s.%s.w-%s.h-%s.%s' % (args.model, args.dataset, args.window, args.horizon, args.rnn_model)

if args.mylog:
    tensorboard_log_dir = 'tensorboard/%s' % (log_token)
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    writer = SummaryWriter(tensorboard_log_dir)
    shutil.rmtree(tensorboard_log_dir)
    logger.info('tensorboard logging to %s', tensorboard_log_dir)

dI_data_loader = DataBasicLoader(args, dI, load_adj=True)
infected_data_loader = DataBasicLoader(args, infected)
recovered_data_loader = DataBasicLoader(args, recovered)

model = ColaGNN_STAN(args, dI_data_loader) 

logger.info('model %s', model)
if args.cuda:
        model.cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('#params:',pytorch_total_params)

def evaluate(tag='val', save=False):
        model.eval()
        total = 0.
        n_samples = 0.
        total_loss = 0.
        y_true, y_pred = [], []
        batch_size = args.batch
        y_pred_mx = []
        y_true_mx = []
        for dI, I, R in zip(dI_data_loader.get_batches(dI_data_loader.val if tag == 'val' else dI_data_loader.test, batch_size),
                                infected_data_loader.get_batches(infected_data_loader.val if tag == 'val' else infected_data_loader.test, batch_size),
                                recovered_data_loader.get_batches(recovered_data_loader.val if tag == 'val' else recovered_data_loader.test, batch_size)):
                X, Y = dI[0], dI[1]
                I_x, I_y = I[0], I[1]
                R_x, R_y = R[0], R[1]
                output, I_hat, R_hat = model(X, I_x, R_x)
                loss_train = F.l1_loss(output, Y[:, -1, :]) # mse_loss
                loss_sir = F.l1_loss(I_hat, I_y) + F.l1_loss(R_hat, R_y) # SIR loss
                total_loss += loss_train.item() + loss_sir.item()
                n_samples += (output.size(0) * dI_data_loader.m)

                y_true_mx.append(Y[:, -1, :].data.cpu())
                y_pred_mx.append(output.data.cpu())

        y_pred_mx = torch.cat(y_pred_mx)
        y_true_mx = torch.cat(y_true_mx) # [n_samples, 47] 

        y_true_states = y_true_mx.numpy() * (dI_data_loader.max - dI_data_loader.min ) * 1.0 + dI_data_loader.min  
        y_pred_states = y_pred_mx.numpy() * (dI_data_loader.max - dI_data_loader.min ) * 1.0 + dI_data_loader.min  #(#n_samples, 47)
    
        # save prediction for the test datset
        if save:
                # result_path = f'result/{args.dataset}/{args.model}/{args.horizon}'
                result_path = f'result/{args.dataset}/{args.window}/{args.model}/{args.horizon}'
                print(result_path)
                if not os.path.exists(result_path):
                        os.makedirs(result_path)
                        y_pred = pd.DataFrame(y_pred_states) # convert to a dataframe
                        y_pred.to_csv(result_path + "/pred.csv", index=False) # save to file
                        y_true = pd.DataFrame(y_true_states)
                        y_true.to_csv(result_path + "/true.csv", index=False)

        rmse_states = np.mean(np.sqrt(mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values'))) # mean of 47
        raw_mae = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
        std_mae = np.std(raw_mae) # Standard deviation of MAEs for all states/places 
        pcc_tmp = []
        for k in range(dI_data_loader.m):
                pcc_tmp.append(pearsonr(y_true_states[:,k],y_pred_states[:,k])[0])
        pcc_states = np.mean(np.array(pcc_tmp)) 
        r2_states = np.mean(r2_score(y_true_states, y_pred_states, multioutput='raw_values'))
        var_states = np.mean(explained_variance_score(y_true_states, y_pred_states, multioutput='raw_values'))

        # convert y_true & y_pred to real data
        y_true = np.reshape(y_true_states,(-1))
        y_pred = np.reshape(y_pred_states,(-1))
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        pcc = pearsonr(y_true,y_pred)[0]
        r2 = r2_score(y_true, y_pred,multioutput='uniform_average') #variance_weighted 
        var = explained_variance_score(y_true, y_pred, multioutput='uniform_average')
        peak_mae = peak_error(y_true_states.copy(), y_pred_states.copy(), dI_data_loader.peak_thold)
        global y_true_t
        global y_pred_t
        y_true_t = y_true_states
        y_pred_t = y_pred_states
        return float(total_loss / n_samples), mae ,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae

def train():
        model.train()
        total_loss = 0.
        n_samples = 0.
        batch_size = args.batch

        for dI, I, R in zip(dI_data_loader.get_batches(dI_data_loader.train, batch_size),
                        infected_data_loader.get_batches(infected_data_loader.train, batch_size),
                        recovered_data_loader.get_batches(recovered_data_loader.train, batch_size)):
                X, Y = dI[0], dI[1]
                I_x, I_y = I[0], I[1]
                R_x, R_y = R[0], R[1]
                optimizer.zero_grad()
                output, I_hat, R_hat = model(X, I_x, R_x)
                loss_train = F.l1_loss(output, Y[:, -1, :]) # mse_loss
                loss_sir = F.l1_loss(I_hat, I_y) + F.l1_loss(R_hat, R_y) # SIR loss
                loss = loss_train + loss_sir
                total_loss += loss_train.item() + loss_sir.item()
                loss.backward()
                optimizer.step()
                n_samples += (output.size(0) * dI_data_loader.m)
        return float(total_loss / n_samples)

# Training loop
bad_counter = 0
best_epoch = 0
best_val = 1e+20
try:
        print('begin training');
        if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
        
        for epoch in range(1, args.epochs+1):
                epoch_start_time = time.time()
                train_loss = train()
                val_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae = evaluate(tag='val')
                print('Epoch {:3d}|time:{:5.2f}s|train_loss {:5.8f}|val_loss {:5.8f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss))

                if args.mylog:
                        writer.add_scalars('data/loss', {'train': train_loss}, epoch )
                        writer.add_scalars('data/loss', {'val': val_loss}, epoch)
                        writer.add_scalars('data/mae', {'val': mae}, epoch)
                        writer.add_scalars('data/rmse', {'val': rmse_states}, epoch)
                        writer.add_scalars('data/rmse_states', {'val': rmse_states}, epoch)
                        writer.add_scalars('data/pcc', {'val': pcc}, epoch)
                        writer.add_scalars('data/pcc_states', {'val': pcc_states}, epoch)
                        writer.add_scalars('data/R2', {'val': r2}, epoch)
                        writer.add_scalars('data/R2_states', {'val': r2_states}, epoch)
                        writer.add_scalars('data/var', {'val': var}, epoch)
                        writer.add_scalars('data/var_states', {'val': var_states}, epoch)
                        writer.add_scalars('data/peak_mae', {'val': peak_mae}, epoch)
        
                # Save the model if the validation loss is the best we've seen so far.
                if val_loss < best_val:
                        best_val = val_loss
                        best_epoch = epoch
                        bad_counter = 0
                        model_path = '%s/%s.pt' % (args.save_dir, log_token)
                        with open(model_path, 'wb') as f:
                                torch.save(model.state_dict(), f)
                                print('Best validation epoch:',epoch, time.ctime());
                                test_loss, mae ,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae = evaluate(tag='test')
                                print('TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}'.format(mae, std_mae, rmse, rmse_states, pcc, pcc_states,r2, r2_states, var, var_states, peak_mae))
                else:
                        bad_counter += 1

                if bad_counter == args.patience:
                        break

except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early, epoch',epoch)

# Load the best saved model.
model_path = '%s/%s.pt' % (args.save_dir, log_token)
with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f));
test_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae = evaluate(tag='test', save=True)
print('Final evaluation')
print('TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}'.format(mae, std_mae, rmse, rmse_states, pcc, pcc_states,r2, r2_states, var, var_states, peak_mae))

with open("run_log.txt", 'a') as f:
        f.write(log_token)
        f.write(': ')
        f.write('TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}'.format(mae, std_mae, rmse, rmse_states, pcc, pcc_states,r2, r2_states, var, var_states, peak_mae))
        f.write('\n\n')

