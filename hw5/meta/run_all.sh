#!/usr/bin/env bash

#################
### Problem 1 ###
#################
py train_policy.py pm-obs --exp_name meta_pmobs_h1_n100_tasks4 --history 1 -lr 5e-5 -n 100 --num_tasks 4

py plot.py data/ data/ --value AverageReturn

#################
### Problem 2 ###
#################
py train_policy.py pm-obs --exp_name meta_pmobs_h60_n100_tasks4_rec --history 60 -lr 5e-5 -n 100 --num_tasks 4 --recurrent
py train_policy.py pm-obs --exp_name meta_pmobs_h30_n100_tasks4_rec --history 30 -lr 5e-5 -n 100 --num_tasks 4 --recurrent
py train_policy.py pm-obs --exp_name meta_pmobs_h120_n100_tasks4_rec --history 120 -lr 5e-5 -n 100 --num_tasks 4 --recurrent
py train_policy.py pm-obs --exp_name meta_pmobs_h60_n100_tasks4 --history 60 -lr 5e-5 -n 100 --num_tasks 4
py train_policy.py pm-obs --exp_name meta_pmobs_h30_n100_tasks4 --history 30 -lr 5e-5 -n 100 --num_tasks 4
py train_policy.py pm-obs --exp_name meta_pmobs_h120_n100_tasks4 --history 120 -lr 5e-5 -n 100 --num_tasks 4

py plot.py data/ data/ --value AverageReturn

#################
### Problem 3 ###
#################
py train_policy.py pm-obs --exp_name <experiment_name> --history 60 -lr 5e-5 -n 100 --recurrent
py train_policy.py pm-obs --exp_name <experiment_name> --history 60 -lr 5e-5 -n 100

py plot.py data/ data/ --value AverageReture --value ValAverageReturn
