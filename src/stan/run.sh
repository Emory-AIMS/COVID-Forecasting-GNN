#! /bin/bash

# echo -e "ColaGNN_STAN_NoAttn_Identity:\n" >> run_log.txt

# nohup python3 train.py --model colagnn_stan_noattn_identity --horizon 2 --dataset "ca48-548" --sim_mat "ca48-adj"
# nohup python3 train.py --model colagnn_stan_noattn_identity --horizon 5 --dataset "ca48-548" --sim_mat "ca48-adj" 
# nohup python3 train.py --model colagnn_stan_noattn_identity --horizon 7 --dataset "ca48-548" --sim_mat "ca48-adj" 
# nohup python3 train.py --model colagnn_stan_noattn_identity --horizon 14 --dataset "ca48-548" --sim_mat "ca48-adj" 
# nohup python3 train.py --model colagnn_stan_noattn_identity --horizon 28 --dataset "ca48-548" --sim_mat "ca48-adj" 

# echo -e "ColaGNN_STAN_NoAttn_SVI:\n" >> run_log.txt

# nohup python3 train.py --model colagnn_stan_noattn_svi --horizon 2 --dataset "ca48-548" --sim_mat "ca48-svi"
# nohup python3 train.py --model colagnn_stan_noattn_svi --horizon 5 --dataset "ca48-548" --sim_mat "ca48-svi" 
# nohup python3 train.py --model colagnn_stan_noattn_svi --horizon 7 --dataset "ca48-548" --sim_mat "ca48-svi" 
# nohup python3 train.py --model colagnn_stan_noattn_svi --horizon 14 --dataset "ca48-548" --sim_mat "ca48-svi" 
# nohup python3 train.py --model colagnn_stan_noattn_svi --horizon 28 --dataset "ca48-548" --sim_mat "ca48-svi" 