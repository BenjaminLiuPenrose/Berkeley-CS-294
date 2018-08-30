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
import pandas as pd

'''===================================================================================================
File content:
Implement compute_stats, compaute useful stats mean reards, std rewards and percentage of full steps
==================================================================================================='''
def compute_stats(data):
	"""
	Arguments:
	data 	-- np.array, input data

	Returns:
	res 	-- pd.sSeries, {"mean reward": ,"std reward": , "pct full steps": }
	"""
	# Preparation Phrase
	res = {}

	# Handling Phrase
	mean = data['returns'].mean()
	std = data['returns'].std()
	steps = data['steps']
	full_rollouts = steps.max()
	pct_full_rollouts = (steps/full_rollouts).mean()

	# Checking Phrase
	res = pd.Series({
		'mean reward': mean,
		'std reward': std,
		'pct full rollouts': pct_full_rollouts
		})
	return res
