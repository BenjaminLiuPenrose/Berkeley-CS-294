#!/usr/bin/env bash

python train_policy.py 'pm-obs' --exp_name <experiment_name> --history 1 -lr 5e-5 -n 100 --num_tasks 4

