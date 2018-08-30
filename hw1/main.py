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
from main_q2 import *
from main_q3 import *

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

logger = logging.getLogger();
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

hdlr = logging.FileHandler(os.path.join(CURRENT_PATH, 'logging_output', "{}.log".format(CURRENT_TIME)));
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
console=logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

logger.setLevel(logging.INFO)

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

def main():
	logging.info("\nHello Sir, welcome to Benjamin's solution to CS294-112 hw1.........................\n")
	main_q2();
	main_q3();
	logging.info("\nSir, congratulations! The program finished successfully, you can find the desired files in...")
	logging.info("\n{}".format(os.path.join(CURRENT_PATH, 'report_output', CURRENT_TIME)))
	logging.info("\nPlease check and have a nice day!")

if __name__=='__main__':
	main()
