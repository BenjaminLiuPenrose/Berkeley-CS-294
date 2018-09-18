#!/bin/bash

set -eux
py train_pg_f18.py LunarLanderContinuous-v2 -ep 1000 --discount 0.99 -n 100 -e 3 -l 2 -s 64 -b 40000 -lr 0.005 -rtg --nn_baseline --exp_name ll_b40000_r0.005

py plot.py data/ll_b40000_r0.005_LunarLanderContinuous-v2_15-09-2018_21-41-36 --value AverageReturn
