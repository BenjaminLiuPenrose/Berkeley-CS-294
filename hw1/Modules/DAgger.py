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
TRAINING_OPTS_DA = glb.TRAINING_OPTS_DA
from Modules.build_neurual_net import *
from Modules.compute_stats import *
from Modules.run_expert_mod import *
from Modules.run_neurual_net_with_D import *
from Modules.run_expert_with_obs import *
from Modules.Tools import *
from sklearn.utils import shuffle
import pandas as pd
'''===================================================================================================
File content:
Implement dataset aggregation

Goal: let data from p_pi instead of p_data

Steps:
1) train pi_theta(neurual net) from D -- this step is the same as behavioral cloning
2) run pi_theta(neurual net) to get D_pi -- run_neurual_net_with_D.py
3) label D_pi with a_t				-- run_expert_with_obs.py
4) Aggregate D<-D U D_pi and repeat ...
==================================================================================================='''
@Timer
def DAgger(config):
	"""
	Arguments:
	config 	-- user configuration
	Returns:
	df 		-- pd.DataFrame
	generating .csv .pkl files
	"""
	# Preparation Phrase
	stats, rewards = {}, {}
	iters = config['iters']

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

	for i in range(iters): #############################################################################
		# step 1: train pi_theta (neurual net model) from D
		o, a = shuffle(o, a)
		model.fit(o, a, **training_opts)

		# step 2: run pi_theta(neurual net) to get D_pi
		dat_o_pi = run_neurual_net_with_D(model, env, config)

		o_new = dat_o_pi['observations']
		stats[i+1] = compute_stats(dat_o_pi)
		rewards[i+1] = dat_o_pi['returns']

		# step 3: label D_pi with a_t
		a_new = run_expert_with_obs(o_new, policy_fn)

		# step 4: Aggregate D<-D U D_pi and repeat
		o = np.append(o, o_new, axis = 0)
		a = np.append(a, a_new.reshape(-1, actions_dim), axis = 0)


	# Checking Phrase
	df = pd.DataFrame(stats).T
	df.index.name = 'iters'
	csv_fname = os.path.join(CURRENT_PATH, 'DAgger_output', CURRENT_TIME, '{}-DAgger-stats.csv'.format(env_name))
	df.to_csv(csv_fname)
	csv_fname = os.path.join(CURRENT_PATH, 'report_output', CURRENT_TIME, '{}-DAgger-stats.csv'.format(env_name))
	df.to_csv(csv_fname)
	pickle_fname = os.path.join(CURRENT_PATH, 'DAgger_output', CURRENT_TIME, '{}-DAgger-rewards.pkl'.format(env_name))
	pickle.dump(rewards, open(pickle_fname, 'wb'))
	logging.info("DAgger.py: finished successfully!")

	return df


def load_policy_fn(env_name):
	policy_fname = os.path.join(CURRENT_PATH, 'experts', '{}.pkl'.format(env_name))
	policy_fn = load_policy.load_policy(policy_fname)
	return policy_fn
