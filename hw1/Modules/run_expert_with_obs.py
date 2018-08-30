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

'''===================================================================================================
File content:
Implement **Step 3) label D_pi with a_t** in **DAgger.py**
==================================================================================================='''
def run_expert_with_obs(observations, policy_fn, env=None, config=None):
	"""
	Arguments:
	observations 	-- observations
	policy_fn 		-- policy function
	env 	-- environment
	config 	-- configuration

	Returns:
	res 	-- np.array, np.array(actions)
	"""
	# Preparation Phrase
	res = np.array([])
	with tf.Session():
		tf_util.initialize()

		actions = []

		# Handling Phrase
		for obs in observations:
			action = policy_fn(obs[None, :])
			actions.append(action)

		# Checking Phrase
		res = np.array(actions)
		return res

