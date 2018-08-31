# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
Name: Beier (Benjamin) Liu
Date: 8/26/2018

Remark:
Python 3.6 is recommended
Before running please install packages *numpy *gym==0.10.5 *mujoco-py==1.50.1.56 *tensorflow==1.5 *seaborn
Using cmd line py -3.6 -m pip install [package_name]
'''
import os, time, logging
import copy, math
import functools, itertools
import tensorflow as tf
import tf_util
import gym
import load_policy
import pickle
import numpy as np

import Modules.glb as glb
CURRENT_TIME = glb.CURRENT_TIME
CURRENT_PATH = glb.CURRENT_PATH
ENV_NAME = glb.ENV_NAME
EXPERT_POLICY_FILE = glb.EXPERT_POLICY_FILE
MAX_TIMESTEPS = glb.MAX_TIMESTEPS
RENDER_EXPERT = glb.RENDER_EXPERT
RENDER_MODEL = glb.RENDER_MODEL
NUM_ROLLOUTS_EXPERT = glb.NUM_ROLLOUTS_EXPERT
NUM_ROLLOUTS_MODEL = glb.NUM_ROLLOUTS_MODEL
LEARNING_RATE_BC = glb.LEARNING_RATE_BC
LEARNING_RATE_DA = glb.LEARNING_RATE_DA
TRAINING_OPTS_DA = glb.TRAINING_OPTS_DA
TRAINING_OPTS_BC = glb.TRAINING_OPTS_BC
ITERS = glb.ITERS
ENV_NAME_THREE = glb.ENV_NAME_THREE
from Modules.DAgger import *
'''===================================================================================================
File content:
run DAgger
==================================================================================================='''
def run_DAgger(config=None):
	"""
	Arguments:
	confid 	-- dict, a dict of user configuration
	Returns:
	res 	-- dict, a dict of useful output
	"""
	res = {}
	num_experts = len(ENV_NAME)

	for i in range(num_experts):
		if ENV_NAME[i] not in ENV_NAME_THREE:
			continue
		config = {
			"expert_policy_file": EXPERT_POLICY_FILE[i],
			"env_name": ENV_NAME[i],
			"render_model": RENDER_MODEL,
			"render_expert": RENDER_EXPERT,
			"max_timesteps": MAX_TIMESTEPS,
			"num_rollouts_expert": NUM_ROLLOUTS_EXPERT,
			"num_rollouts_model": NUM_ROLLOUTS_MODEL,
			'learning_rate': LEARNING_RATE_DA,
			'training_opts': TRAINING_OPTS_DA,
			'iters': ITERS
		}

		res['plain test']=DAgger(config)

	return res


