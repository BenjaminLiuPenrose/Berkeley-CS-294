#!/usr/bin/env bash

#################
### Problem 1 ###
#################
py train_policy.py pm-obs --exp_name meta_pmobs_h1_n100_tasks4 --history 1 -lr 5e-5 -n 100 --num_tasks 4

#################
### Problem 2 ###
#################
py train_policy.py pm-obs --exp_name <experiment_name> --history 60 -lr 5e-5 -n 100 --num_tasks 4 --recurrent
py train_policy.py pm-obs --exp_name <experiment_name> --history 60 -lr 5e-5 -n 100 --num_tasks 4


#################
### Problem 3 ###
#################
py train_policy.py pm-obs --exp_name <experiment_name> --history 1 -lr 5e-5 -n 100
