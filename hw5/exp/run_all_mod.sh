#!/usr/bin/env bash

##########################
### P1 Hist PointMass  ###
##########################

py train_ac_exploration_f18.py PointMass-v0 -n 100 -b 1000 -e 3 --density_model none -s 8 --exp_name PM_bc0_s8
py train_ac_exploration_f18.py PointMass-v0 -n 100 -b 1000 -e 3 --density_model hist -bc 0.01 -s 8 --exp_name PM_hist_bc0.01_s8

py plot.py data/ac_PM_bc0_s8_PointMass-v0_07-11-2018_22-48-12 data/ac_PM_hist_bc0.01_s8_PointMass-v0_07-11-2018_22-49-59 --value AverageReturn

##########################
###  P2 RBF PointMass  ###
##########################

py train_ac_exploration_f18.py PointMass-v0 -n 100 -b 1000 -e 3 --density_model rbf -bc 0.01 -s 8 -sig 0.2 --exp_name PM_rbf_bc0.01_s8_sig0.2

py plot.py data/ac_PM_bc0_s8_PointMass-v0_07-11-2018_22-48-12 data/ac_PM_rbf_bc0.01_s8_sig0.2_PointMass-v0_07-11-2018_22-54-41 --value AverageReturn

##########################
###  P3 EX2 PointMass  ###
##########################

py train_ac_exploration_f18.py PointMass-v0 -n 100 -b 1000 -e 3 --density_model ex2 -s 8 -bc 0.05 -kl 0.1 -dlr 0.001 -dh 8 --exp_name PM_ex2_s8_bc0.05_kl0.1_dlr0.001_dh8

py plot.py data/ac_PM_bc0_s8_PointMass-v0_07-11-2018_22-48-12 data/ac_PM_ex2_s8_bc0.05_kl0.1_dlr0.001_dh8_PointMass-v0_08-11-2018_00-35-55 --value AverageReturn

###########################
###    P4 HalfCheetah   ###
###########################

py train_ac_exploration_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 8 -l 2 -s 32 -b 30000 -lr 0.02 --density_model none --exp_name HC_bc0
py train_ac_exploration_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 8 -l 2 -s 32 -b 30000 -lr 0.02 --density_model ex2 -bc 0.001 -kl 0.1 -dlr 0.005 -dti 1000 --exp_name HC_bc0.001_kl0.1_dlr0.005_dti1000
py train_ac_exploration_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 8 -l 2 -s 32 -b 30000 -lr 0.02 --density_model ex2 -bc 0.0001 -kl 0.1 -dlr 0.005 -dti 10000 --exp_name HC_bc0.0001_kl0.1_dlr0.005_dti10000

# py plot.py data/ac_HC_bc0_HalfCheetah-v2_08-11-2018_00-47-12 data/ac_HC_bc0.001_kl0.1_dlr0.005_dti10000_HalfCheetah-v2_08-11-2018_02-33-55 data/ac_HC_bc0.001_kl0.1_dlr0.005_dti1000_HalfCheetah-v2_08-11-2018_01-42-01 --value AverageReturn

# py plot.py data/ac_HC_bc0_HalfCheetah-v2_08-11-2018_23-23-21 data/ac_HC_bc0.001_kl0.1_dlr0.005_dti10000_HalfCheetah-v2_09-11-2018_03-22-45 data/ac_HC_bc0.001_kl0.1_dlr0.005_dti1000_HalfCheetah-v2_09-11-2018_01-20-35 --value AverageReturn

py plot.py data/ac_HC_bc0_HalfCheetah-v2_10-11-2018_02-41-14 data/ac_HC_bc0.001_kl0.1_dlr0.005_dti1000_HalfCheetah-v2_10-11-2018_04-21-43 data/ac_HC_bc0.0001_kl0.1_dlr0.005_dti10000_HalfCheetah-v2_10-11-2018_06-29-15 --value AverageReturn

