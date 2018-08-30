# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
Name: Beier (Benjamin) Liu
Date: 8/27/2018

Remark:
Python 3.6 is recommended
Before running please install packages *numpy *gym==0.10.5 *mujoco-py==1.50.1.56 *tensorflow==1.5 *seaborn
Using cmd line py -3.6 -m pip install [package_name]
'''
import os, time, logging
import copy, math
import functools, itertools
import Modules.glb as glb
CURRENT_TIME = glb.CURRENT_TIME
CURRENT_PATH = glb.CURRENT_PATH
ENV_NAME = glb.ENV_NAME
from Modules.run_DAgger import *
import pandas as pd

if not os.path.exists(os.path.join(CURRENT_PATH, 'expert_data')):
	os.makedirs(os.path.join(CURRENT_PATH, 'expert_data'))
if not os.path.exists(os.path.join(CURRENT_PATH, 'expert_output')):
	os.makedirs(os.path.join(CURRENT_PATH, 'expert_output'))
if not os.path.exists(os.path.join(CURRENT_PATH, 'behavioral_cloning_output', CURRENT_TIME)):
	os.makedirs(os.path.join(CURRENT_PATH, 'behavioral_cloning_output', CURRENT_TIME))
if not os.path.exists(os.path.join(CURRENT_PATH, 'DAgger_output', CURRENT_TIME)):
	os.makedirs(os.path.join(CURRENT_PATH, 'DAgger_output', CURRENT_TIME))
if not os.path.exists(os.path.join(CURRENT_PATH, 'logging_output')):
	os.makedirs(os.path.join(CURRENT_PATH, 'logging_output'))
if not os.path.exists(os.path.join(CURRENT_PATH, 'report_output', CURRENT_TIME)):
	os.makedirs(os.path.join(CURRENT_PATH, 'report_output', CURRENT_TIME))


'''===================================================================================================
File content:
Run the problem sec 2 and sec 3

Section 3. DAgger
3.1  Implement DAgger. See the code provided in run expert.py to see how to query the expert policy
and perform roll-outs in the environment.

3.2 Run DAgger and report results on one task in which DAgger can learn a better policy than behavioral
cloning. Report your results in the form of a learning curve, plotting the number of DAgger iterations
vs. the policy’s mean return, with error bars to show the standard deviation. Include the performance
of the expert policy and the behavioral cloning agent on the same plot. In the caption, state which
task you used, and any details regarding network architecture, amount of data, etc. (as in the previous
section).
==================================================================================================='''

def main_q3():
	logging.info("\n==============================solving section 3 =======================================\n")
	res = {}
	logging.info("\n==============================running question 2.2======================================\n")
	res['plain_test'] = run_DAgger()

	logging.info("\n==============================summarizing stats for section 3======================================\n")
	logging.info("This is done by combing *[expert_name]-DAgger-stats.csv.* and *[expert_name]-BC-stats-ep.csv]* in folder report_output")
	plot_q3()

def plot_q3(kword="ep", path=os.path.join(CURRENT_PATH, 'report_output', CURRENT_TIME)):
	import seaborn as sns; sns.set()
	import matplotlib.pyplot as plt
	env_name =ENV_NAME
	for expert_name in env_name:
		csv_fname_DA = os.path.join(path, '{}-DAgger-stats.csv'.format(expert_name))
		csv_fname_BC = os.path.join(path, '{}-BC-stats-{}.csv'.format(expert_name, kword))
		df_DA = pd.read_csv(csv_fname_DA)
		df_BC = pd.read_csv(csv_fname_BC, index_col=0)
		x_bc, bc_mean_reward, bc_std_reward, expert_mean_reward, expert_std_reward = df_BC.iloc[:,0], df_BC.iloc[:, 1], df_BC.iloc[:, 2], df_BC.iloc[:, 3], df_BC.iloc[:, 4]
		# x_label = df_DA.columns.tolist()[0]
		x_da, da_mean_reward, da_std_reward = df_DA.iloc[:,0], df_DA.iloc[:, 1], df_DA.iloc[:, 2]

		fig, ax = plt.subplots()
		clrs = sns.color_palette("husl", 3)
		ax.plot(x_da, da_mean_reward, label="DAgger", c=clrs[0])
		ax.fill_between(x_da, da_mean_reward-da_std_reward, da_mean_reward+da_std_reward, alpha=0.3, facecolor=clrs[0])
		ax.plot(x_bc, bc_mean_reward, label="BehavioralCloning", c=clrs[1])
		ax.fill_between(x_bc, bc_mean_reward-bc_std_reward, bc_mean_reward+bc_std_reward, alpha=0.3, facecolor=clrs[1])
		ax.plot(x_bc, expert_mean_reward, label="ExpertPolicy", c=clrs[2])
		ax.fill_between(x_bc, expert_mean_reward-expert_std_reward, expert_mean_reward+expert_std_reward, alpha=0.3, facecolor=clrs[2])
		ax.legend(loc='best')
		plt.xlabel("iters (for DAgger) or epochs (for BC and Expert)")
		plt.ylabel("rewards")
		plt.title("rewards vs iters/epochs | {}".format(expert_name))
		png_fname = os.path.join(path, '{}-DAgger-rewards.png'.format(expert_name))
		plt.savefig(png_fname)
		plt.close()

if __name__=='__main__':
	"""======================================================================================================
	if want to run the program normally, comment out the rest, leave main_q3() alone
	"""
	main_q3()

	"""========================================================================================================
	if you want to plot for question 3.2, leave plot_q3(path=os.path.join(CURRENT_PATH, "report_output", "YOUR NAME HERE")) only, comment out the rest
	and you need to copy the [exprt_name]-DAgger-stats.csv file to the dir = os.path.join(CURRENT_PATH, "report_output", "YOUR NAME HERE")

	or

	copy all the needed csv files to ~..report_output, leave plot_q3(path=os.path.join(CURRENT_PATH, "report_output") only, comment out the rest
	and you need to copy the [exprt_name]-DAgger-stats.csv file to the dir = os.path.join(CURRENT_PATH, "report_output")

	again rewards png pics will be in ~..report_output
	"""
	# plot_q3(path=os.path.join(CURRENT_PATH, "report_output", "YOUR NAME HERE"))
	# plot_q3(path=os.path.join(CURRENT_PATH, "report_output"))
