#!/bin/bash

set -eux
py train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -dna --exp_name sb_no_rtg_dna
py train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna --exp_name sb_rtg_dna
py train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg --exp_name sb_rtg_na
py train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -dna --exp_name lb_no_rtg_dna
py train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna --exp_name lb_rtg_dna
py train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg --exp_name lb_rtg_na

py plot.py data/sb_no_rtg_dna_CartPole-v0_12-09-2018_00-23-34 data/sb_rtg_dna_CartPole-v0_15-09-2018_19-28-58 data/sb_rtg_na_CartPole-v0_15-09-2018_19-44-28 --value AverageReturn
py plot.py data/lb_no_rtg_dna_CartPole-v0_15-09-2018_19-45-55 data/lb_rtg_dna_CartPole-v0_15-09-2018_19-52-24 data/lb_rtg_na_CartPole-v0_15-09-2018_19-57-13 --value AverageReturn
