#!/bin/bash
# py -m pip install tensorflow==1.5
set -eux
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2
do
    py run_expert.py experts/$e.pkl $e --render --num_rollouts=1
done
