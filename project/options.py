# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse

def get_options(option_type):
  """
  option_type: string
    'training' or 'display' or 'visualize'
  """
  parser = argparse.ArgumentParser()
  # Common
  parser.add_argument("--env_type", type= str, default="gym", choices=("gym", "maze", "lab"), help="environment type (lab or gym or maze)")
  parser.add_argument("--env_name", type=str, default="PongNoFrameskip-v4",  help="environment name")
  parser.add_argument("--use_pixel_change", type=bool, default=True, help="whether to use pixel change")
  parser.add_argument("--use_value_replay", type=bool, default=True, help="whether to use value function replay")
  parser.add_argument("--use_reward_prediction", type=bool, default=True, help="whether to use reward prediction")

  parser.add_argument("--checkpoint_dir", type=str, default="/tmp/unreal_checkpoints", help="checkpoint directory")

  # For training
  if option_type == 'training':
    parser.add_argument("--parallel_size", type=int, default=8, help="parallel thread size")
    parser.add_argument("--local_t_max", type=int, default=20, help="repeat step size")
    parser.add_argument("--rmsp_alpha", type=float, default=0.99, help="decay parameter for rmsprop")
    parser.add_argument("--rmsp_epsilon", type=float, default=0.1, help="epsilon parameter for rmsprop")

    parser.add_argument("--log_file", type=str, default="/tmp/unreal_log/unreal_log", help="log file directory")
    parser.add_argument("--initial_alpha_low", type=float, default=1e-4, help="log_uniform low limit for learning rate")
    parser.add_argument("--initial_alpha_high", type=float, default=5e-3, help="log_uniform high limit for learning rate")
    parser.add_argument("--initial_alpha_log_rate", type=float, default=0.5, help="log_uniform interpolate rate for learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor for rewards")
    parser.add_argument("--gamma_pc", type=float, default=0.9, help="discount factor for pixel control")
    parser.add_argument("--entropy_beta", type=float, default=0.001, help="entropy regularization constant")
    parser.add_argument("--pixel_change_lambda", type=float, default=0.001, help="pixel change lambda") # 0.05, 0.01 ~ 0.1 for lab, 0.0001 ~ 0.01 for gym
    parser.add_argument("--value_replay_lambda", type=float, default=1.0, help="value replay lambda")
    parser.add_argument("--reward_prediction_lambda", type=float, default=1.0, help="reward prediction lambda")
    parser.add_argument("--experience_history_size", type=int, default=2000, help="experience replay buffer size")
    parser.add_argument("--max_time_step", type=int, default=10 * 10**7, help="max time steps")
    parser.add_argument("--save_interval_step", type=int, default=100 * 1000, help="saving interval steps")
    parser.add_argument("--grad_norm_clip", type=float, default=40.0, help="gradient norm clipping")

  # For display
  if option_type == 'display':
    parser.add_argument("--frame_save_dir", type=str, deault="/tmp/unreal_frames", help="frame save directory")
    parser.add_argument("--recording", type=bool, default=False, help="whether to record movie")
    parser.add_argument("--frame_saving", type=bool, default=False, help="whether to save frames")

  args = parser.parse_args()
  return args

# def get_options(option_type):
#   """
#   option_type: string
#     'training' or 'display' or 'visualize'
#   """
#   # Common
#   tf.app.flags.DEFINE_string("env_type", "lab", "environment type (lab or gym or maze)")
#   tf.app.flags.DEFINE_string("env_name", "nav_maze_static_01",  "environment name")
#   tf.app.flags.DEFINE_boolean("use_pixel_change", True, "whether to use pixel change")
#   tf.app.flags.DEFINE_boolean("use_value_replay", True, "whether to use value function replay")
#   tf.app.flags.DEFINE_boolean("use_reward_prediction", True, "whether to use reward prediction")

#   tf.app.flags.DEFINE_string("checkpoint_dir", "/tmp/unreal_checkpoints", "checkpoint directory")

#   # For training
#   if option_type == 'training':
#     tf.app.flags.DEFINE_integer("parallel_size", 8, "parallel thread size")
#     tf.app.flags.DEFINE_integer("local_t_max", 20, "repeat step size")
#     tf.app.flags.DEFINE_float("rmsp_alpha", 0.99, "decay parameter for rmsprop")
#     tf.app.flags.DEFINE_float("rmsp_epsilon", 0.1, "epsilon parameter for rmsprop")

#     tf.app.flags.DEFINE_string("log_file", "/tmp/unreal_log/unreal_log", "log file directory")
#     tf.app.flags.DEFINE_float("initial_alpha_low", 1e-4, "log_uniform low limit for learning rate")
#     tf.app.flags.DEFINE_float("initial_alpha_high", 5e-3, "log_uniform high limit for learning rate")
#     tf.app.flags.DEFINE_float("initial_alpha_log_rate", 0.5, "log_uniform interpolate rate for learning rate")
#     tf.app.flags.DEFINE_float("gamma", 0.99, "discount factor for rewards")
#     tf.app.flags.DEFINE_float("gamma_pc", 0.9, "discount factor for pixel control")
#     tf.app.flags.DEFINE_float("entropy_beta", 0.001, "entropy regularization constant")
#     tf.app.flags.DEFINE_float("pixel_change_lambda", 0.05, "pixel change lambda") # 0.05, 0.01 ~ 0.1 for lab, 0.0001 ~ 0.01 for gym
#     tf.app.flags.DEFINE_integer("experience_history_size", 2000, "experience replay buffer size")
#     tf.app.flags.DEFINE_integer("max_time_step", 10 * 10**7, "max time steps")
#     tf.app.flags.DEFINE_integer("save_interval_step", 100 * 1000, "saving interval steps")
#     tf.app.flags.DEFINE_boolean("grad_norm_clip", 40.0, "gradient norm clipping")

#   # For display
#   if option_type == 'display':
#     tf.app.flags.DEFINE_string("frame_save_dir", "/tmp/unreal_frames", "frame save directory")
#     tf.app.flags.DEFINE_boolean("recording", False, "whether to record movie")
#     tf.app.flags.DEFINE_boolean("frame_saving", False, "whether to save frames")

#   return tf.app.flags.FLAGS
