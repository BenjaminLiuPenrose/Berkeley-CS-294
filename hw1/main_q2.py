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
import os, time, logging, glob
import copy, math
import functools, itertools
import Modules.glb as glb
CURRENT_TIME = glb.CURRENT_TIME
CURRENT_PATH = glb.CURRENT_PATH
ENV_NAME = glb.ENV_NAME
from Modules.run_behavioral_cloning import *
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

Section 2. Behavioral Cloning
2.1 The starter code provides an expert policy for each of the MuJoCo tasks in OpenAI Gym (See
run expert.py). Generate roll-outs from the provided policies, and implement behavioral cloning.

2.2  Run behavioral cloning (BC) and report results on two tasks – one task where a behavioral cloning
agent achieves comparable performance to the expert, and one task where it does not. When providing
results, report the mean and standard deviation of the return over multiple rollouts in a table, and state
which task was used. Be sure to set up a fair comparison, in terms of network size, amount of data,
and number of training iterations, and provide these details (and any others you feel are appropriate)
in the table caption.

2.3 Experiment with one hyperparameter that affects the performance of the behavioral cloning agent, such
as the number of demonstrations, the number of training epochs, the variance of the expert policy, or
something that you come up with yourself. For one of the tasks used in the previous question, show a
graph of how the BC agent’s performance varies with the value of this hyperparameter, and state the
hyperparameter and a brief rationale for why you chose it in the caption for the graph.
==================================================================================================='''

def main_q2():
	logging.info("\n==============================solving section 2 =======================================\n")
	res = {}
	logging.info("\n==============================running question 2.2======================================\n")
	config = {'boolean_plain_test': True}
	res['plain_test'] = run_behavioral_cloning(config)

	logging.info("\n==============================running question 2.3======================================\n")
	logging.info("\n==============================testing hyperparam learning rate======================================\n")
	config = {
		'boolean_learning_rate_test': True,
		'boolean_epochs_test': False,
		'boolean_num_rollouts_model_test': False
	}
	res['learning_rate_test'] = run_behavioral_cloning(config)


	logging.info("\n==============================testing hyperparam epochs======================================\n")
	config = {
		'boolean_learning_rate_test': False,
		'boolean_epochs_test': True,
		'boolean_num_rollouts_model_test': False
	}
	res['epochs_test'] = run_behavioral_cloning(config)

	logging.info("\n==============================summarizing stats for section 2======================================\n")
	kwords = ["md", "lr", "ep"]
	summarize_stats_from_csv(kwords)
	plot_q2_after_summarize_stats(["lr", "ep"])


def summarize_stats_from_csv(kwords, extension="csv", path=os.path.join(CURRENT_PATH, "behavioral_cloning_output", CURRENT_TIME)):
	import re
	os.chdir(path)
	files_ls = {}
	for kword in kwords:
		files_ls[kword] = [i for i in glob.glob('{}-*.{}'.format(kword, extension))]
	os.chdir(CURRENT_PATH)

	env_name = ENV_NAME
	for kword in kwords:
		if kword=="md":
			df = pd.DataFrame(columns=["expert name", "expert mean reward", "expert std reward", "BC mean reward", "BC std reward"])
			cnt=0
			for f in files_ls[kword]:
				expert_name = re.search('-(.*)-v2', f).group(1) + "-v2"
				dat = pd.read_csv(os.path.join(path, f))
				df.loc[cnt] = [expert_name, dat.iloc[0, 1], dat.iloc[0, 2], dat.iloc[1, 1], dat.iloc[1, 2]]
				cnt+=1
			csv_fname = os.path.join(CURRENT_PATH, 'report_output', CURRENT_TIME, 'BC-stats-summary.csv')
			df.index.name = '#'
			df.to_csv(csv_fname)
		elif kword=="lr":
			for expert_name in env_name:
				df = pd.DataFrame(columns=["learning rate", "BC mean reward", "BC std reward", "expert mean reward", "expert std reward"])
				cnt = 0
				for f in files_ls[kword]:
					if expert_name in f:
						lr = float(re.search('learning_rate-(.*).csv', f).group(1))
						dat = pd.read_csv(os.path.join(path, f))
						df.loc[cnt] = [lr, dat.iloc[1, 1], dat.iloc[1, 2], dat.iloc[0, 1], dat.iloc[0, 2]]
						cnt+=1
				csv_fname = os.path.join(CURRENT_PATH, 'report_output', CURRENT_TIME, '{}-BC-stats-{}.csv'.format(expert_name, kword))
				df.sort_values('learning rate', inplace=True)
				df.set_index(np.arange(len(df.index)), inplace=True)
				df.index.name = '#'
				df.to_csv(csv_fname)
		elif kword=="ep":
			for expert_name in env_name:
				df = pd.DataFrame(columns=["epochs", "BC mean reward", "BC std reward", "expert mean reward", "expert std reward"])
				cnt = 0
				for f in files_ls[kword]:
					if expert_name in f:
						epochs = int(re.search('epochs-(.*).csv', f).group(1))
						dat = pd.read_csv(os.path.join(path, f))
						df.loc[epochs] = [epochs, dat.iloc[1, 1], dat.iloc[1, 2], dat.iloc[0, 1], dat.iloc[0, 2]]
						cnt+=1
				csv_fname = os.path.join(CURRENT_PATH, 'report_output', CURRENT_TIME, '{}-BC-stats-{}.csv'.format(expert_name, kword))
				df.sort_values('epochs', inplace=True)
				df.set_index(np.arange(len(df.index)), inplace=True)
				df.index.name = '#'
				df.to_csv(csv_fname)
		else:
			for expert_name in env_name:
				df = pd.DataFrame(columns=["num rollouts", "BC mean reward", "BC std reward", "expert mean reward", "expert std reward"])
				for f in files_ls:
					if expert_name in f:
						num_rollouts = int(re.search('num_rollouts-(.*).csv', f).group(1))
						dat = pd.read_csv(os.path.join(path, f))
						df.loc[num_rollouts] = [num_rollouts, dat.iloc[1, 1], dat.iloc[1, 2], dat.iloc[0, 1], dat.iloc[0, 2]]
				csv_fname = os.path.join(CURRENT_PATH, 'report_output', CURRENT_TIME, '{}-BC-stats-{}.csv'.format(expert_name,
					kword))
				df.sort_values('num rollouts', inplace=True)
				df.set_index(np.arange(len(df.index)), inplace=True)
				df.index.name = '#'
				df.to_csv(csv_fname)

def plot_q2_after_summarize_stats(kwords, path=os.path.join(CURRENT_PATH, 'report_output', CURRENT_TIME)):
	import seaborn as sns; sns.set()
	import matplotlib.pyplot as plt
	env_name =ENV_NAME
	for expert_name in env_name:
		for kword in kwords:
			csv_fname = os.path.join(path, '{}-BC-stats-{}.csv'.format(expert_name, kword))
			df = pd.read_csv(csv_fname, index_col=0)
			x_kword, bc_mean_reward, bc_std_reward, expert_mean_reward, expert_std_reward = df.iloc[:,0], df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 3], df.iloc[:, 4]
			x_label = df.columns.tolist()[0]

			fig, ax = plt.subplots()
			clrs = sns.color_palette("husl", 3)
			ax.plot(x_kword, bc_mean_reward, label="BehavioralCloning", c=clrs[1])
			ax.fill_between(x_kword, bc_mean_reward-bc_std_reward, bc_mean_reward+bc_std_reward, alpha=0.3, facecolor=clrs[1])
			ax.plot(x_kword, expert_mean_reward, label="ExpertPolicy", c=clrs[2])
			ax.fill_between(x_kword, expert_mean_reward-expert_std_reward, expert_mean_reward+expert_std_reward, alpha=0.3, facecolor=clrs[2])
			# ax = sns.lineplot(x=x_kword, y=bc_mean_reward, color='blue', legend='brief', linestyle='-')
			# sns.lineplot(x=x_kword, y=expert_mean_reward, color='red', legend='brief', linestyle='--')
			# ax.errorbar(x_kword, bc_mean_reward, yerr=bc_std_reward, fmt='-p')
			# ax.errorbar(x_kword, expert_mean_reward, yerr=expert_std_reward, fmt='-o')
			if x_label=="learning rate":
				ax.set_xscale('log')
			ax.legend(loc='best')
			plt.xlabel(x_label)
			plt.ylabel("rewards")
			plt.title("rewards vs {} | {}".format(x_label ,expert_name))
			png_fname = os.path.join(path, '{}-BC-rewards-{}.png'.format(expert_name, kword))
			plt.savefig(png_fname)
			plt.close()

if __name__=='__main__':
	"""=====================================================================================================
	if want to run the program normally, comment out the rest, leave main_q2() alone
	"""
	main_q2()

	"""=========================================================================================================
	if want to summarize the stats of question 2 only, leave summarize_stats_from_csv(kwords=["md", "lr", "ep"], path=os.path.join(CURRENT_PATH, "behavioral_cloning_output", "YOUR NAME HERE") only, commet out the rest

	or

	copy all the needed csv files to ~..behavioral_cloning_output, leave summarize_stats_from_csv(kwords=["md", "lr", "ep"], path=os.path.join(CURRENT_PATH, "behavioral_cloning_output") only, commet out the rest

	again summary csv files will be in ~..report_output
	"""
	# summarize_stats_from_csv(kwords=["md", "lr", "ep"], path=os.path.join(CURRENT_PATH, "behavioral_cloning_output", "YOUR NAME HERE"))
	# summarize_stats_from_csv(kwords=["md", "lr", "ep"], path=os.path.join(CURRENT_PATH, "behavioral_cloning_output"))

	""" ==================================================================================================
	if you want to plot for question 2.3, leave plot_q2_after_summarize_stats(["lr", "ep"], path=os.path.join(CURRENT_PATH, "report_output", "YOUR NAME HERE")) only, comment out the rest

	or

	copy all the needed csv files to ~..report_output, leave plot_q2_after_summarize_stats(["lr", "ep"], path=os.path.join(CURRENT_PATH, "report_output")) only, comment out the rest

	again rewards png pics will be in ~..report_output
	"""
	# plot_q2_after_summarize_stats(["lr", "ep"], path=os.path.join(CURRENT_PATH, "report_output", "YOUR NAME HERE"))
	# plot_q2_after_summarize_stats(["lr", "ep"], path=os.path.join(CURRENT_PATH, "report_output"))
