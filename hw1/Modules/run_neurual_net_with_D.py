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
Implement **Step 2) run pi_theta(neurual net) to get D_pi** in **DAgger.py**
also used by behavioral_cloning.py
==================================================================================================='''
def run_neurual_net_with_D(model, env, config):
	"""
	Arguments:
	model 	-- trained nn model
	env 	-- the training environment and settings
	config 	-- user configuration

	Returns:
	res 	-- dict of np.array, {"observations": , "actions": , "returns": , "steps": }
	"""
	# Preparation Phrase
	res = {}
	max_steps = env.spec.timestep_limit
	num_rollouts = config['num_rollouts_model']
	render = config['render_model']

	returns = []
	observations = []
	actions = []
	num_steps = []

	# Checking Phrase
	for i in range(num_rollouts):
		obs = env.reset()
		done = False
		totalr = 0.
		steps = 0
		while not done:
			action = model.predict(obs[None, :])
			observations.append(obs)
			actions.append(action)
			obs, r, done, _ = env.step(action)
			totalr += r
			steps += 1
			if render:
				env.render()
			if steps > max_steps:
				logging.info("run_neurual_net_with_D.py: step exceeds max_steps {}".format(max_steps))
				break
		num_steps.append(steps)
		returns.append(totalr)

	# Handling Phrase
	logging.info("run_neurual_net_with_D.py: finished successfully!")
	res = {
		"observations": np.array(observations),
		"actions": np.array(actions),
		"returns": np.array(returns),
		"steps": np.array(num_steps)
	}
	return res
