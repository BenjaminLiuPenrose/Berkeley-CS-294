# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
Name: Beier (Benjamin) Liu
Date: 8/27/2018

Remark:
Python 3.6 is recommended
Before running please install packages *numpy
Using cmd line py -3.6 -m pip install [package_name]
'''
import os, time, logging
import copy, math
import functools, itertools
import numpy as np
import datetime

'''===================================================================================================
File content:
global variables of the whole program

==================================================================================================='''

CURRENT_PATH = os.getcwd();
CURRENT_TIME = datetime.datetime.now().strftime('%Y%m%d_%H%M%S');

ENV_NAME =  [
	"Reacher-v2",
	"Ant-v2",
	"HalfCheetah-v2",
	"Hopper-v2",
	"Humanoid-v2",
	"Walker2d-v2"
];
EXPERT_POLICY_FILE = [os.path.join(CURRENT_PATH, 'experts', x+".pkl") for x in ENV_NAME]
MAX_TIMESTEPS = None ## this variable is never used
RENDER_EXPERT = False
RENDER_MODEL = False
NUM_ROLLOUTS_EXPERT = 20
NUM_ROLLOUTS_MODEL = 20

### NN settings
LEARNING_RATE_BC = 0.001
EPOCHS = 10

LEARNING_RATE_DA = 0.001


TRAINING_OPTS_DA = dict(
	validation_split=0.1,
	batch_size=256,
	epochs=10,
	verbose=2
)

TRAINING_OPTS_BC = dict(
	validation_split=0.1,
	batch_size=256,
	epochs=EPOCHS,
	verbose=2
)

#### Hyperparams tuning
LEARNING_RATE_BC_LIST = [0.01*(10**(-x/2)) for x in range(9)]+[1e-6];
EPOCHS_LIST = [1]+[5*(x+1) for x in range(8)]+[50];
NUM_ROLLOUTS_MODEL_LIST = [1]+[5*(x+1) for x in range(8)]+[50];

### DAgger settings
ITERS = 50

### Files flag mapping
MAP_DICT = {
	"md": "model",
	"lr": "learning rate",
	"ep": "epochs",
	"ro": "num rollouts"
}


ENV_NAME =  [
	"Humanoid-v2"
];

LEARNING_RATE_BC_LIST = [0.01]+[1e-6];
EPOCHS_LIST = [1]+[50];
NUM_ROLLOUTS_MODEL_LIST = [1]+[50];
