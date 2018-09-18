#!/bin/bash

set -eux
py train_pg_f18.py Walker2d-v2 -ep 150 --discount 0.99 -n 100 -e 3 -l 2 -s 32 -b 5000 -lr 1e-2 -rtg -bl --exp_name wa_b5000_r1e-2

py train_pg_f18.py Walker2d-v2 -ep 150 --discount 0.99 -n 100 -e 3 -l 2 -s 32 -b 5000 -lr 1e-2 -rtg -bl --gae --lamda 0.9 --exp_name wa_b5000_r1e-2_d9

py train_pg_f18.py Walker2d-v2 -ep 150 --discount 0.99 -n 100 -e 3 -l 2 -s 32 -b 5000 -lr 1e-2 -rtg -bl --gae --lamda 0.99 --exp_name wa_b5000_r1e-2_d99


py plot.py data/wa_b5000_r1e-2_Walker2d-v2_17-09-2018_14-22-45 data/wa_b5000_r1e-2_d99_Walker2d-v2_17-09-2018_15-05-23 --value AverageReturn

py plot.py data/wa_b5000_r1e-2_Walker2d-v2_17-09-2018_14-22-45 data/wa_b5000_r1e-2_d9_Walker2d-v2_17-09-2018_14-57-32 --value AverageReturn

# ========================================OLD AND DEPRECIATED METHODS===================================================
# py plot.py data/wa_b500_r1e-2_null_Walker2d-v2_17-09-2018_14-06-39 data/wa_b500_r1e-2_Walker2d-v2_17-09-2018_14-07-46 --value AverageReturn

# py train_pg_f18.py Walker2d-v2 -ep 150 --discount 0.99 -n 100 -e 3 -l 2 -s 32 -b 5000 -lr 1e-2 --exp_name wa_b5000_r1e-2_null

