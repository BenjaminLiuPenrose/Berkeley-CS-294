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

from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras.optimizers import Adam

'''===================================================================================================
File content:
Implement building (but not training) neurural net
==================================================================================================='''
def build_neurual_net(data, env, config):
	"""
	Arguments:
	data	-- data to be trained on the neurual nets
	env 	-- simulation env
	config 	-- config of cmd/console

	Returns:
	model	-- a untrained built neurual net model
	"""
	# Preparation Phrase
	learning_rate = config['learning_rate']
	mean = np.mean(data['observations'], axis = 0)
	std = np.std(data['observations'], axis = 0) + 1e-6 ############################################################

	observations_dim = env.observation_space.shape[0]
	actions_dim = env.action_space.shape[0]

	# Handling Phrase
	model = Sequential([
		Lambda(lambda x: (x-mean)/std, batch_input_shape=(None, observations_dim)),
	Dense(64, activation='tanh'), #############################################################################
	Dense(64, activation='tanh'),
	Dense(actions_dim)
	])

	opt = Adam(lr=learning_rate)
	model.compile(optimizer=opt, loss='mse', metrics=['mse'])

	# Checking Phrase
	return model
