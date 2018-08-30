#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

'''===================================================================================================
File content:
run export policy (for num_rollouts)
==================================================================================================='''
def run_expert_mod(policy_fn, env, config):
    """
    Arguments:
    policy_fn   -- policy function
    env     -- traing env
    config  -- dict, a dict of user configuration
    Returns:
    res     -- dict, a dict useful info
    """
    res = {}
    env_name = config['env_name']
    max_timesteps = config['max_timesteps']
    render = config['render_expert']
    num_rollouts = config['num_rollouts_expert']

    with tf.Session():
        tf_util.initialize()

        # env = gym.make(env_name)
        max_steps = max_timesteps or env.spec.timestep_limit
        max_steps = env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        num_steps = []

        for i in range(num_rollouts):
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None, :])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps > max_steps:
                    logging.info("run_expert_mod.py: step exceeds max_steps {}".format(max_steps))
                    break
            num_steps.append(steps)
            returns.append(totalr)

        res = {
            "observations": np.array(observations),
            "actions": np.array(actions),
            "returns": np.array(returns),
            "steps": np.array(num_steps)
        }

        if False:
            with open(os.path.join('expert_output', env_name + '.pkl'), 'wb') as f:
                pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)

        return res



        # # Handling Phrase
        # for i in range(args.num_rollouts):
        #     print('iter', i)
        #     obs = env.reset()
        #     done = False
        #     totalr = 0.
        #     steps = 0
        #     while not done:
        #         action = policy_fn(obs[None,:])
        #         observations.append(obs) # record observation
        #         actions.append(action) # record action
        #         obs, r, done, _ = env.step(action)
        #         totalr += r # total returns is acc rewards
        #         steps += 1
        #         if args.render:
        #             env.render()
        #         if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
        #         if steps >= max_steps:
        #             break
        #     returns.append(totalr) # record total returns
