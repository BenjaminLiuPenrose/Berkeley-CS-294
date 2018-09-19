#!/bin/bash

set -eux
py train_pg_f18.py Walker2d-v2 -ep 150 --discount 0.99 -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 2e-2 -rtg -bl --exp_name wa_b5000_r1e-2

py train_pg_f18.py Walker2d-v2 -ep 150 --discount 0.99 -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 2e-2 -rtg -bl --gae --lamda 0.9 --exp_name wa_b5000_r1e-2_d9

py train_pg_f18.py Walker2d-v2 -ep 150 --discount 0.99 -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 2e-2 -rtg -bl --gae --lamda 0.99 --exp_name wa_b5000_r1e-2_d99
