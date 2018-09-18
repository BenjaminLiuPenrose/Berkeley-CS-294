#!/bin/bash

set -eux
for batch in 100 500 1000 3000 5000
do
	for lr in 2e-2 1e-2 5e-3 1e-3 1e-4
	do
		py train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b $batch -lr $lr -rtg --exp_name hc_b${batch}_r${lr}
	done
done

py plot.py data/hc_b1000_r1e-2_InvertedPendulum-v2_16-09-2018_00-45-21 data/hc_b100_r1e-2_InvertedPendulum-v2_16-09-2018_00-33-44 data/hc_b3000_r1e-2_InvertedPendulum-v2_16-09-2018_00-54-45 data/hc_b5000_r1e-2_InvertedPendulum-v2_16-09-2018_01-11-36 data/hc_b500_r1e-2_InvertedPendulum-v2_16-09-2018_00-38-46 --value AverageReturn

py plot.py data/hc_b1000_r1e-3_InvertedPendulum-v2_16-09-2018_00-48-48 data/hc_b100_r1e-3_InvertedPendulum-v2_16-09-2018_00-35-48 data/hc_b3000_r1e-3_InvertedPendulum-v2_16-09-2018_01-00-40 data/hc_b5000_r1e-3_InvertedPendulum-v2_16-09-2018_01-26-49 data/hc_b500_r1e-3_InvertedPendulum-v2_16-09-2018_00-41-37 --value AverageReturn

py plot.py data/hc_b1000_r1e-4_InvertedPendulum-v2_16-09-2018_00-50-12 data/hc_b100_r1e-4_InvertedPendulum-v2_16-09-2018_00-36-35 data/hc_b3000_r1e-4_InvertedPendulum-v2_16-09-2018_01-03-28 data/hc_b5000_r1e-4_InvertedPendulum-v2_16-09-2018_01-35-03 data/hc_b500_r1e-4_InvertedPendulum-v2_16-09-2018_00-42-33 --value AverageReturn

py plot.py data/hc_b1000_r2e-2_InvertedPendulum-v2_16-09-2018_00-43-32 data/hc_b100_r2e-2_InvertedPendulum-v2_16-09-2018_00-32-34 data/hc_b3000_r2e-2_InvertedPendulum-v2_16-09-2018_00-51-39 data/hc_b5000_r2e-2_InvertedPendulum-v2_16-09-2018_01-06-47 data/hc_b500_r2e-2_InvertedPendulum-v2_16-09-2018_00-37-17 --value AverageReturn

py plot.py data/hc_b1000_r5e-3_InvertedPendulum-v2_16-09-2018_00-47-05 data/hc_b100_r5e-3_InvertedPendulum-v2_16-09-2018_00-34-49 data/hc_b3000_r5e-3_InvertedPendulum-v2_16-09-2018_00-57-44 data/hc_b5000_r5e-3_InvertedPendulum-v2_16-09-2018_01-17-25 data/hc_b500_r5e-3_InvertedPendulum-v2_16-09-2018_00-40-14 --value AverageReturn

py plot.py data/hc_b500_r1e-2_InvertedPendulum-v2_16-09-2018_00-38-46 --value AverageReturn

# =========================================================OLD AND DEPRECIATED===========================================
# =======================================================================================================================
# =======================================================================================================================
# py plot.py data/hc_b1000_r1e-2_InvertedPendulum-v2_16-09-2018_00-45-21 data/hc_b1000_r1e-3_InvertedPendulum-v2_16-09-2018_00-48-48 data/hc_b1000_r1e-4_InvertedPendulum-v2_16-09-2018_00-50-12 data/hc_b1000_r2e-2_InvertedPendulum-v2_16-09-2018_00-43-32 data/hc_b1000_r5e-3_InvertedPendulum-v2_16-09-2018_00-47-05 --value AverageReturn

# py plot.py data/hc_b100_r1e-2_InvertedPendulum-v2_16-09-2018_00-33-44 data/hc_b100_r1e-3_InvertedPendulum-v2_16-09-2018_00-35-48 data/hc_b100_r1e-4_InvertedPendulum-v2_16-09-2018_00-36-35 data/hc_b100_r2e-2_InvertedPendulum-v2_16-09-2018_00-32-34 data/hc_b100_r5e-3_InvertedPendulum-v2_16-09-2018_00-34-49 --value AverageReturn

# py plot.py data/hc_b3000_r1e-2_InvertedPendulum-v2_16-09-2018_00-54-45 data/hc_b3000_r1e-3_InvertedPendulum-v2_16-09-2018_01-00-40 data/hc_b3000_r1e-4_InvertedPendulum-v2_16-09-2018_01-03-28 data/hc_b3000_r2e-2_InvertedPendulum-v2_16-09-2018_00-51-39 data/hc_b3000_r5e-3_InvertedPendulum-v2_16-09-2018_00-57-44 --value AverageReturn

# py plot.py data/hc_b5000_r1e-2_InvertedPendulum-v2_16-09-2018_01-11-36 data/hc_b5000_r1e-3_InvertedPendulum-v2_16-09-2018_01-26-49 data/hc_b5000_r1e-4_InvertedPendulum-v2_16-09-2018_01-35-03 data/hc_b5000_r2e-2_InvertedPendulum-v2_16-09-2018_01-06-47 data/hc_b5000_r5e-3_InvertedPendulum-v2_16-09-2018_01-17-25 --value AverageReturn

# py plot.py data/hc_b500_r1e-2_InvertedPendulum-v2_16-09-2018_00-38-46 data/hc_b500_r1e-3_InvertedPendulum-v2_16-09-2018_00-41-37 data/hc_b500_r1e-4_InvertedPendulum-v2_16-09-2018_00-42-33 data/hc_b500_r2e-2_InvertedPendulum-v2_16-09-2018_00-37-17 data/hc_b500_r5e-3_InvertedPendulum-v2_16-09-2018_00-40-14 --value AverageReturn


