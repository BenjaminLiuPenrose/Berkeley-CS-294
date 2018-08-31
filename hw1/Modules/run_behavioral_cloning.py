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
LEARNING_RATE_BC_LIST = glb.LEARNING_RATE_BC_LIST
EPOCHS_LIST = glb.EPOCHS_LIST
NUM_ROLLOUTS_MODEL_LIST = glb.NUM_ROLLOUTS_MODEL_LIST
ENV_NAME_TWO_LR =  glb.ENV_NAME_TWO_LR
ENV_NAME_TWO_EP = glb.ENV_NAME_TWO_EP
from Modules.behavioral_cloning import *
'''===================================================================================================
File content:
run behavioral cloning
==================================================================================================='''
def run_behavioral_cloning(config={'boolean_plain_test': True}):
	"""
	Arguments:
	config 	-- a dict, user configuration
	Returns:
	res 	-- a dict, useful outputs from BC
	"""
	res = {}
	num_experts = len(ENV_NAME)
	if config.get('boolean_plain_test', False)!=False:
		res['plain_test'] = {}
		for i in range(num_experts):
			conf = {
				"expert_policy_file": EXPERT_POLICY_FILE[i],
				"env_name": ENV_NAME[i],
				"render_model": RENDER_MODEL,
				"render_expert": RENDER_EXPERT,
				"max_timesteps": MAX_TIMESTEPS,
				"num_rollouts_expert": NUM_ROLLOUTS_EXPERT,
				"num_rollouts_model": NUM_ROLLOUTS_MODEL,
				'learning_rate': LEARNING_RATE_BC,
				'training_opts': TRAINING_OPTS_BC,
				'boolean_hyperparams_test': False
			}

			res['plain_test'][ENV_NAME[i]] = behavioral_cloning(conf)

	if config.get('boolean_learning_rate_test', False)!=False:
		res['learning_rate_test'] = {}
		for i in range(num_experts):
			res['learning_rate_test'][ENV_NAME[i]] = {}
			if ENV_NAME[i] not in ENV_NAME_TWO_LR:
				continue
			for lr in LEARNING_RATE_BC_LIST:
				conf = {
					"expert_policy_file": EXPERT_POLICY_FILE[i],
					"env_name": ENV_NAME[i],
					"render_model": RENDER_MODEL,
					"render_expert": RENDER_EXPERT,
					"max_timesteps": MAX_TIMESTEPS,
					"num_rollouts_expert": NUM_ROLLOUTS_EXPERT,
					"num_rollouts_model": NUM_ROLLOUTS_MODEL,
					'learning_rate': lr,
					'training_opts': TRAINING_OPTS_BC,
					'boolean_hyperparams_test': True,
					'name_hyperparams_test': 'learning_rate'
				}

				res['learning_rate_test'][ENV_NAME[i]][lr] = behavioral_cloning(conf)

	if config.get('boolean_epochs_test', False)!=False:
		res['epochs_test'] = {}
		for i in range(num_experts):
			if ENV_NAME[i] not in ENV_NAME_TWO_EP:
				continue
			res['epochs_test'][ENV_NAME[i]] = {}
			for epochs in EPOCHS_LIST:
				training_opts = TRAINING_OPTS_BC
				training_opts['epochs'] = epochs
				conf = {
					"expert_policy_file": EXPERT_POLICY_FILE[i],
					"env_name": ENV_NAME[i],
					"render_model": RENDER_MODEL,
					"render_expert": RENDER_EXPERT,
					"max_timesteps": MAX_TIMESTEPS,
					"num_rollouts_expert": NUM_ROLLOUTS_EXPERT,
					"num_rollouts_model": NUM_ROLLOUTS_MODEL,
					'learning_rate': LEARNING_RATE_BC,
					'training_opts': training_opts,
					'boolean_hyperparams_test': True,
					'name_hyperparams_test': 'epochs',
					'epochs': epochs
				}

				res['epochs_test'][ENV_NAME[i]][epochs] = behavioral_cloning(conf)

	if config.get('boolean_num_rollouts_model_test', False)!=False:
		res['num_rollouts_model'] = {}
		for i in range(num_experts):
			if ENV_NAME[i] not in ENV_NAME_TWO_LR:
				continue
			res['num_rollouts_model'][ENV_NAME[i]] = {}
			for num_rollouts_model in NUM_ROLLOUTS_MODEL_LIST:
				conf = {
					"expert_policy_file": EXPERT_POLICY_FILE[i],
					"env_name": ENV_NAME[i],
					"render_model": RENDER_MODEL,
					"render_expert": RENDER_EXPERT,
					"max_timesteps": MAX_TIMESTEPS,
					"num_rollouts_expert": NUM_ROLLOUTS_EXPERT,
					"num_rollouts_model": num_rollouts_model,
					'learning_rate': LEARNING_RATE_BC,
					'training_opts': TRAINING_OPTS_BC,
					'boolean_hyperparams_test': True,
					'name_hyperparams_test': 'num_rollouts_model'
				}

				res['num_rollouts_model'][ENV_NAME[i]][num_rollouts_model] = behavioral_cloning(conf)

	return res
