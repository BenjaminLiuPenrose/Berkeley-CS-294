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
TRAINING_OPTS_BC = glb.TRAINING_OPTS_BC
from Modules.build_neurual_net import *
from Modules.compute_stats import *
from Modules.run_expert_mod import *
from Modules.run_neurual_net_with_D import *
from Modules.Tools import *
from sklearn.utils import shuffle
import pandas as pd
'''===================================================================================================
File content:
Implement behavioral cloning

Steps:
1) collect the data from expert -- run_expert.py
2) train neurual_net on data -- neurual_net.py
==================================================================================================='''
@Timer
def behavioral_cloning(config):
	"""
	Arguments:
	config 	-- user configuration
	Returns:
	df 		-- a pd.DataFrame
	generating .csv .pkl files
	"""
	# Preparation Phrase
	env_name = config['env_name']
	policy_fn = load_policy_fn(env_name)

	env = gym.make(env_name)
	actions_dim = env.action_space.shape[0]
	training_opts = config['training_opts']

	dat = run_expert_mod(policy_fn, env, config)
	o = dat['observations']
	a = dat['actions'].reshape(-1, actions_dim)

	# Handling Phrase
	model = build_neurual_net(dat, env, config)

	o, a = shuffle(o, a)
	model.fit(o, a, **training_opts)

	dat_bc = run_neurual_net_with_D(model, env, config)

	stats_expert = compute_stats(dat)
	stats_bc = compute_stats(dat_bc)
	stats = {'expert': stats_expert, 'behavioral cloning': stats_bc}

	# Checking Phrase
	df = pd.DataFrame(stats).T

	boolean_hyperparams_test = config.get('boolean_hyperparams_test', False)
	if not boolean_hyperparams_test:
		csv_fname = os.path.join(CURRENT_PATH, 'behavioral_cloning_output', CURRENT_TIME, 'md-{}-BC-stats.csv'.format(env_name))
		df.to_csv(csv_fname)
		pickle_fname = os.path.join(CURRENT_PATH, 'behavioral_cloning_output', CURRENT_TIME, 'md-{}-BC-res.pkl'.format(env_name))
		pickle.dump(dat_bc, open(pickle_fname, 'wb'))
		logging.info("behavioral_cloning.py: finished successfully!")
	else:
		if config.get('name_hyperparams_test', 'NA')=='learning_rate':
			fname_tail = "{}-{:.2g}".format(config.get('name_hyperparams_test', 'NA'), config.get(config.get('name_hyperparams_test', 'NA'), 'NA'))
			fname_head = "lr"
		elif config.get('name_hyperparams_test', 'NA')=='epochs':
			fname_tail = "{}-{}".format(config.get('name_hyperparams_test', 'NA'), config.get(config.get('name_hyperparams_test', 'NA'), 'NA'))
			fname_head = "ep"
		else:
			fname_tail = "{}-{}".format(config.get('name_hyperparams_test', 'NA'), config.get(config.get('name_hyperparams_test', 'NA'), 'NA'))
			fname_head = "ro"
		csv_fname = os.path.join(CURRENT_PATH, 'behavioral_cloning_output', CURRENT_TIME, '{}-{}-BC-stats-{}.csv'.format(fname_head, env_name, fname_tail))
		df.to_csv(csv_fname)
		pickle_fname = os.path.join(CURRENT_PATH, 'behavioral_cloning_output', CURRENT_TIME, '{}-{}-BC-res-{}.pkl'.format(fname_head, env_name, fname_tail))
		pickle.dump(dat_bc, open(pickle_fname, 'wb'))
		logging.info("behavioral_cloning.py: {} finished successfully!".format(fname_tail))

	return df

def load_policy_fn(env_name):
	policy_fname = os.path.join(CURRENT_PATH, 'experts', '{}.pkl'.format(env_name))
	policy_fn = load_policy.load_policy(policy_fname)
	return policy_fn
