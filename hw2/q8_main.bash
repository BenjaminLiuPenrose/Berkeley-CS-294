#!/bin/bash

set -eux
for batch in 10000 30000 50000
do
	for lr in 2e-2 1e-2 5e-3
	do
		py train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b $batch -lr $lr --exp_name hc_b${batch}_r${lr}
	done
done

py plot.py data/hc_b10000_r1e-2_HalfCheetah-v2_16-09-2018_01-34-15 data/hc_b10000_r2e-2_HalfCheetah-v2_16-09-2018_01-15-35 data/hc_b10000_r5e-3_HalfCheetah-v2_16-09-2018_01-48-43 data/hc_b30000_r1e-2_HalfCheetah-v2_16-09-2018_02-29-15 data/hc_b30000_r2e-2_HalfCheetah-v2_16-09-2018_01-58-45 data/hc_b30000_r5e-3_HalfCheetah-v2_16-09-2018_03-02-26 data/hc_b50000_r1e-2_HalfCheetah-v2_16-09-2018_10-17-20 data/hc_b50000_r2e-2_HalfCheetah-v2_16-09-2018_09-29-15 data/hc_b50000_r5e-3_HalfCheetah-v2_16-09-2018_11-35-48 --value AverageReturn

py train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 2e-2 --exp_name hc_b50000_r2e-2

py train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 2e-2 -rtg --exp_name hc_b50000_r2e-2

py train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 2e-2 --nn_baseline --exp_name hc_b50000_r2e-2

py train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 2e-2 -rtg --nn_baseline --exp_name hc_b50000_r2e-2


py plot.py data/hc_b50000_r2e-2_HalfCheetah-v2_17-09-2018_00-48-48 --value AverageReturn

py plot.py data/hc_b50000_r2e-2_HalfCheetah-v2_17-09-2018_01-37-14 --value AverageReturn

py plot.py data/hc_b50000_r2e-2_HalfCheetah-v2_17-09-2018_02-24-07 --value AverageReturn

py plot.py data/hc_b50000_r2e-2_HalfCheetah-v2_17-09-2018_03-11-07 --value AverageReturn
