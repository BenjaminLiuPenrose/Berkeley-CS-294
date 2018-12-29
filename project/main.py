# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import threading
from multiprocessing import Process
import signal
import math
import os
import time

from environment.environment import Environment
from model.model import UnrealModel
from train.trainer import Trainer
from train.rmsprop_applier import RMSPropApplier
from options import get_options


USE_GPU = True # To use GPU, set True

# get command line args
flags = get_options("training")
# flags = get_options("display")

def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1-rate) + log_hi * rate
    return math.exp(v)


class Application(object):
    def __init__(self):
        pass

    # def train_function(self, parallel_index, preparing):
    #     """ Train each environment. """

    #     trainer = self.trainers[parallel_index]
    #     if preparing:
    #         trainer.prepare()

    #     # set start_time
    #     trainer.set_start_time(self.start_time)

    #     # total_timesteps global_t = 0
    #     # for itr in range(n_iter max_time_steps):
    #     while True:
    #         if self.stop_requested: # call method self.save(), only callable from therad-0.
    #             break
    #         if self.terminate_reqested: # press Ctrl+C
    #             trainer.stop()
    #             break
    #         if self.global_t > flags.max_time_step:
    #             trainer.stop()
    #             break
    #         if parallel_index == 0 and self.global_t > self.next_save_steps:
    #             # Save checkpoint
    #             self.save()
    #         # logging rewards
    #         diff_global_t, _ = trainer.process(self.sess,
    #                                                                         self.global_t,
    #                                                                         self.summary_writer,
    #                                                                         self.summary_op,
    #                                                                         self.score_input)
    #         self.global_t += diff_global_t

    def run(self):
        device = "/cpu:0"
        if USE_GPU:
            device = "/gpu:0"

        initial_learning_rate = log_uniform(flags.initial_alpha_low,
                                                                                flags.initial_alpha_high,
                                                                                flags.initial_alpha_log_rate)

        self.global_t = 0

        self.stop_requested = False
        self.terminate_reqested = False

        action_size = Environment.get_action_size(flags.env_type,
                                                                                            flags.env_name)

        self.global_network = UnrealModel(action_size,
                                                                            -1,
                                                                            flags.use_pixel_change,
                                                                            flags.use_value_replay,
                                                                            flags.use_reward_prediction,
                                                                            flags.pixel_change_lambda,
                                                                            flags.value_replay_lambda,
                                                                            flags.reward_prediction_lambda,
                                                                            flags.entropy_beta,
                                                                            device)
        self.trainers = []

        learning_rate_input = tf.placeholder("float")

        grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                                                    decay = flags.rmsp_alpha,
                                                                    momentum = 0.0,
                                                                    epsilon = flags.rmsp_epsilon,
                                                                    clip_norm = flags.grad_norm_clip,
                                                                    device = device)

        for i in range(flags.parallel_size):
            trainer = Trainer(i,
                                                self.global_network,
                                                initial_learning_rate,
                                                learning_rate_input,
                                                grad_applier,
                                                flags.env_type,
                                                flags.env_name,
                                                flags.use_pixel_change,
                                                flags.use_value_replay,
                                                flags.use_reward_prediction,
                                                flags.pixel_change_lambda,
                                                flags.value_replay_lambda,
                                                flags.reward_prediction_lambda,
                                                flags.entropy_beta,
                                                flags.local_t_max,
                                                flags.gamma,
                                                flags.gamma_pc,
                                                flags.experience_history_size,
                                                flags.max_time_step,
                                                device)
            self.trainers.append(trainer)

        # prepare session
        config = tf.ConfigProto(log_device_placement=False,
                                                        allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())

        # summary for tensorboard
        self.score_input = tf.placeholder(tf.int32)
        tf.summary.scalar("score", self.score_input)

        self.summary_op = tf.summary.merge_all()
        # logging rewards, see in tensorboard
        self.summary_writer = tf.summary.FileWriter(flags.log_file,
                                                    self.sess.graph)

        # init or load checkpoint with saver
        self.saver = tf.train.Saver(self.global_network.get_vars())

        checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("DEBUG: checkpoint loaded:", checkpoint.model_checkpoint_path)
            tokens = checkpoint.model_checkpoint_path.split("-")
            # set global step
            self.global_t = int(tokens[1])
            print("DEBUG>>> global step set: ", self.global_t)
            # set wall time
            wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)
            # with open(wall_t_fname, 'r') as f:
            #     self.wall_t = float(f.read())
            self.next_save_steps = (self.global_t + flags.save_interval_step) // flags.save_interval_step * flags.save_interval_step

        else:
            print("DEBUG: Could not find old checkpoint")
            # set wall time
            self.wall_t = 0.0
            self.next_save_steps = flags.save_interval_step

        # run training threads
        self.train_threads = []
        for i in range(flags.parallel_size):
            self.train_threads.append(threading.Thread(target=self.train_function, args=(i,True)))

        signal.signal(signal.SIGINT, self.signal_handler)

        # set start time
        self.start_time = time.time() - self.wall_t

        for t in self.train_threads:
            t.start()

        print('DEBUG: Press Ctrl+C to stop')
        signal.pause()

    def save(self):
        """ Save checkpoint.
        Called from therad-0.
        """
        # logging
        self.stop_requested = True

        # Wait for all other threads to stop
        for (i, t) in enumerate(self.train_threads):
            if i != 0:
                t.join()

        # Save
        if not os.path.exists(flags.checkpoint_dir):
            os.mkdir(flags.checkpoint_dir)

        # Write wall time
        wall_t = time.time() - self.start_time
        wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)
        # print(206)
        # with open(wall_t_fname, 'w') as f:
        #     print("write wall time")
        #     print(209)
        #     f.write(str(wall_t))
        #     print(211)

        print('Start saving.')
        self.saver.save(self.sess,
                                        flags.checkpoint_dir + '/' + 'checkpoint',
                                        global_step = self.global_t)
        print('End saving.')

        self.stop_requested = False
        self.next_save_steps += flags.save_interval_step

        # Restart other threads
        for i in range(flags.parallel_size):
            if i != 0:
                thread = threading.Thread(target=self.train_function, args=(i,False))
                self.train_threads[i] = thread
                thread.start()

    def signal_handler(self, signal, frame):
        # print('You pressed Ctrl+C!')
        self.terminate_reqested = True

    #######################################################################################################################
    ########################   Local Implementation #######################################################################
    #######################################################################################################################

    ################
    ### Zhaoxin ####
    ################
    def meta_run(self):
        # flags.pixel_change_lambda, flags.value_replay_lambda, flags.reward_prediction_lambda
        weight_meta = [1,1,1]
        step_0 = 10000
        step_1 = 100000
        num_iter_meta = int(flags.max_time_step/step_1)
        for iters in range(num_iter_meta):
            print("=====================================================current iteration: ",iters, "==============================================")
            self.total_reward = []
            self.run_train(True,False,False,weight_meta[0]/1000,weight_meta[1],weight_meta[2],False,True,step_1*iters+step_0)
            self.run_train(False,True,False,weight_meta[0]/1000,weight_meta[1],weight_meta[2],False,True,step_1*iters+step_0)
            self.run_train(False,False,True,weight_meta[0]/1000,weight_meta[1],weight_meta[2],False,True,step_1*iters+step_0)
            # print(self.total_reward.index(max(self.total_reward)))
            max_idx = self.total_reward.index(max(self.total_reward))
            min_idx = self.total_reward.index(min(self.total_reward))
            weight_meta[max_idx] *= 1.05
            weight_meta[min_idx] *= 0.95
            self.run_train(True,True,True,weight_meta[0]/100,weight_meta[1],weight_meta[2],True,True,step_1*(iters+1))

    ###############
    ### Zhaoxin ###
    ###############

    def run_train(self,pc,vr,rp,pc_w,vr_w,rp_w,save,load_cp,step):
        self.current_reward = 0
        self.mod_run(pc,vr,rp,pc_w,vr_w,rp_w,save,load_cp,step)
        # print("5")
        for (i, t) in enumerate(self.train_threads):
            print("thread:", i)
            if i != 0:
                t.join()
        # print("6")
        self.total_reward.append(self.current_reward)

    def mod_run(self,pc,vr,rp,pc_w,vr_w,rp_w,save,load_cp,step):
        device = "/cpu:0"
        if USE_GPU:
            device = "/gpu:0"

        initial_learning_rate = log_uniform(flags.initial_alpha_low,
                                            flags.initial_alpha_high,
                                            flags.initial_alpha_log_rate)

        self.global_t = 0

        self.stop_requested = False
        self.terminate_reqested = False

        action_size = Environment.get_action_size(flags.env_type,
                                                  flags.env_name)

        self.global_network = UnrealModel(action_size,
                                          -1,
                                          pc,
                                          vr,
                                          rp,
                                          pc_w,
                                          vr_w,
                                          rp_w,
                                          flags.entropy_beta,
                                          device)
        self.trainers = []
        learning_rate_input = tf.placeholder("float")

        grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                      decay = flags.rmsp_alpha,
                                      momentum = 0.0,
                                      epsilon = flags.rmsp_epsilon,
                                      clip_norm = flags.grad_norm_clip,
                                      device = device)

        for i in range(flags.parallel_size):
            trainer = Trainer(i,
                            self.global_network,
                            initial_learning_rate,
                            learning_rate_input,
                            grad_applier,
                            flags.env_type,
                            flags.env_name,
                            pc,
                            vr,
                            rp,
                            pc_w,
                            vr_w,
                            rp_w,
                            flags.entropy_beta,
                            flags.local_t_max,
                            flags.gamma,
                            flags.gamma_pc,
                            flags.experience_history_size,
                            flags.max_time_step,
                            device)
            self.trainers.append(trainer)

        # prepare session
        config = tf.ConfigProto(log_device_placement=False,
                                allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())

        # summary for tensorboard
        self.score_input = tf.placeholder(tf.int32)
        tf.summary.scalar("score", self.score_input)

        self.summary_op = tf.summary.merge_all()
        # logging rewards, see in tensorboard
        self.summary_writer = tf.summary.FileWriter(flags.log_file,
                                                    self.sess.graph)

        # init or load checkpoint with saver
        self.saver = tf.train.Saver(self.global_network.get_vars())

        checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path and load_cp:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("DEBUG: checkpoint loaded:", checkpoint.model_checkpoint_path)
            tokens = checkpoint.model_checkpoint_path.split("-")
            # set global step
            self.global_t = int(tokens[1])
            print("DEBUG>>> global step set: ", self.global_t)
            # set wall time
            wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)

            ################
            ### Benjamin ###
            ################
            self.wall_t = 0.0
            # print(364)
            # with open(wall_t_fname, 'r') as f:
            #     print(366)
            #     self.wall_t = float(f.read())
            self.next_save_steps = (self.global_t + flags.save_interval_step) // flags.save_interval_step * flags.save_interval_step

        else:
            print("DEBUG: Could not find old checkpoint")
            # set wall time
            self.wall_t = 0.0
            self.next_save_steps = flags.save_interval_step

        # run training threads
        self.train_threads = []
        for i in range(flags.parallel_size):
            self.train_threads.append(threading.Thread(target=self.train_function, args=(i,True,save,step)))

        #signal.signal(signal.SIGINT, self.signal_handler)

        # set start time
        self.start_time = time.time() - self.wall_t

        for t in self.train_threads:
            t.start()
        # print("after start")
        #signal.pause()

        for t in (self.train_threads):
            print("thread:", i)
            t.join()

        ###############
        ### Zhaoxin ###
        ###############
        tf.reset_default_graph()
        self.sess.close()

    ###############
    ### Zhaoxin ###
    ###############
    def train_function(self, parallel_index, preparing,save,step):
        """ Train each environment. """

        trainer = self.trainers[parallel_index]
        if preparing:
            trainer.prepare()

        # set start_time
        trainer.set_start_time(self.start_time)

        # total_timesteps global_t = 0
        # for itr in range(n_iter max_time_steps):
        while True:
            if self.stop_requested: # call method self.save(), only callable from therad-0.
                break
            if self.terminate_reqested: # press Ctrl+C
                trainer.stop()
                break
            if self.global_t > step:
                # print("1")
                # if parallel_index != 0:
                #     trainer.stop()
                if parallel_index == 0:
                    self.stop_requested =True
                    # for (i, t) in enumerate(self.train_threads):
                    #     print("thread:", i)
                    #     if i != 0:
                    #         t.join()
                    if save:
                        self.mod_save()
                break
            if parallel_index == 0 and self.global_t > self.next_save_steps and save:
                # Save checkpoint
                self.save()
            # logging rewards
            #print("before process")
            diff_global_t,total_reward = trainer.process(self.sess,
                                            self.global_t,
                                            self.summary_writer,
                                            self.summary_op,
                                            self.score_input)
            self.global_t += diff_global_t
            self.current_reward += total_reward

        # print("2")
        trainer.stop()

    ###############
    ### Benjamin ##
    ###############
    def mod_save(self):
        """ Save checkpoint.
        Called from therad-0.
        """
        # logging
        self.stop_requested = True

        # Save
        if not os.path.exists(flags.checkpoint_dir):
          os.mkdir(flags.checkpoint_dir)

        # Write wall time
        wall_t = time.time() - self.start_time
        wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)

        # with open(wall_t_fname, 'w') as f:
        #     print(469)
        #     print(self.global_t)
        #     f.write(str(wall_t))
        #     print(471)

        print('Start saving.')
        self.saver.save(self.sess,
                        flags.checkpoint_dir + '/' + 'checkpoint',
                        global_step = self.global_t)
        print('End saving.')

        self.stop_requested = False
        self.next_save_steps += flags.save_interval_step

        # # Restart other threads
        # for i in range(flags.parallel_size):
        #   if i != 0:
        #     thread = threading.Thread(target=self.train_function, args=(i,False))
        #     self.train_threads[i] = thread
        #     thread.start()

def main(argv):
    app = Application()
    app.meta_run()
    # app.run()

if __name__ == '__main__':
    tf.app.run()
