#!/usr/bin/env bash
# RECALL http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Applications_files/asyncrl.pdf
# parser.add_argument("--env_type", type= str, default="gym", choices=("gym", "maze", "lab"), help="environment type (lab or gym or maze)")
# parser.add_argument("--env_name", type=str, default="PongNoFrameskip-v4",  help="environment name")
# parser.add_argument("--use_pixel_change", type=bool, default=True, help="whether to use pixel change")
# parser.add_argument("--use_value_replay", type=bool, default=True, help="whether to use value function replay")
# parser.add_argument("--use_reward_prediction", type=bool, default=True, help="whether to use reward prediction")
# parser.add_argument("--checkpoint_dir", type=str, default="/tmp/unreal_checkpoints", help="checkpoint directory") # logging tf checkpoints

# if option_type == 'training':
#   parser.add_argument("--parallel_size", type=int, default=8, help="parallel thread size") # parallel workers number threads number
#   parser.add_argument("--local_t_max", type=int, default=20, help="repeat step size")
#   parser.add_argument("--rmsp_alpha", type=float, default=0.99, help="decay parameter for rmsprop")
#   parser.add_argument("--rmsp_epsilon", type=float, default=0.1, help="epsilon parameter for rmsprop")

#   parser.add_argument("--log_file", type=str, default="/tmp/unreal_log/unreal_log", help="log file directory") # logging rewards
#   parser.add_argument("--initial_alpha_low", type=float, default=1e-4, help="log_uniform low limit for learning rate")
#   parser.add_argument("--initial_alpha_high", type=float, default=5e-3, help="log_uniform high limit for learning rate")
#   parser.add_argument("--initial_alpha_log_rate", type=float, default=0.5, help="log_uniform interpolate rate for learning rate")
#   parser.add_argument("--gamma", type=float, default=0.99, help="discount factor for rewards")
#   parser.add_argument("--gamma_pc", type=float, default=0.9, help="discount factor for pixel control")
#   parser.add_argument("--entropy_beta", type=float, default=0.001, help="entropy regularization constant")
#   parser.add_argument("--pixel_change_lambda", type=float, default=0.001, help="pixel change lambda") # 0.05, 0.01 ~ 0.1 for lab, 0.0001 ~ 0.01 for gym
#   parser.add_argument("--experience_history_size", type=int, default=2000, help="experience replay buffer size")
#   parser.add_argument("--max_time_step", type=int, default=5 * 10**6, help="max time steps") # num_iters
#   parser.add_argument("--save_interval_step", type=int, default=100 * 1000, help="saving interval steps")
#   parser.add_argument("--grad_norm_clip", type=float, default=40.0, help="gradient norm clipping")

# if option_type == 'display':
#   parser.add_argument("--frame_save_dir", type=str, deault="/tmp/unreal_frames", help="frame save directory") # plot videoframe
#   parser.add_argument("--recording", type=bool, default=False, help="whether to record movie")
#   parser.add_argument("--frame_saving", type=bool, default=False, help="whether to save frames")

##########################
### P1   ###
##########################
python3 main.py --env_type gym --env_name {}NoFrameskip-v4 --use_pixel_change True --use_value_replay True --use_reward_prediction True --max_time_step {}000000 --checkpoint_dir /tmp/{}_checkpoints --log_file /tmp/{}_log
tensorboard --logdir=/tmp/{}_log

python3 main.py --env_type gym --env_name AlienNoFrameskip-v4 --use_pixel_change True --use_value_replay True --use_reward_prediction True --max_time_step 8000000 --checkpoint_dir /tmp/alien_checkpoints --log_file /tmp/alien_log
tensorboard --logdir=/tmp/alien_log

python3 main.py --env_type gym --env_name AmidarNoFrameskip-v4 --use_pixel_change True --use_value_replay True --use_reward_prediction True --max_time_step 10000000 --checkpoint_dir /tmp/Amidar_checkpoints --log_file /tmp/Amidar_log
tensorboard --logdir=/tmp/Amidar_log

python3 main.py --env_type gym --env_name AssaultNoFrameskip-v4 --use_pixel_change True --use_value_replay True --use_reward_prediction True --max_time_step 10000000 --checkpoint_dir /tmp/Assault_checkpoints --log_file /tmp/Assault_log
tensorboard --logdir=/tmp/Assault_log

python3 main.py --env_type gym --env_name AsterixNoFrameskip-v4 --use_pixel_change True --use_value_replay True --use_reward_prediction True --max_time_step 10000000 --checkpoint_dir /tmp/Asterix_checkpoints --log_file /tmp/Asterix_log
tensorboard --logdir=/tmp/Asterix_log

# =====================================================================================
python3 main.py --env_type gym --env_name PongNoFrameskip-v4 --use_pixel_change False --use_value_replay False --use_reward_prediction False --max_time_step 10000000 --checkpoint_dir /tmp/Pong_fff_checkpoints --log_file /tmp/Pong_fff_log
tensorboard --logdir=/tmp/Pong_fff_log
python3 main.py --env_type gym --env_name PongNoFrameskip-v4 --use_pixel_change True --use_value_replay False --use_reward_prediction False --max_time_step 10000000 --checkpoint_dir /tmp/Pong_tff_checkpoints --log_file /tmp/Pong_tff_log
tensorboard --logdir=/tmp/Pong_tff_log
python3 main.py --env_type gym --env_name PongNoFrameskip-v4 --use_pixel_change False --use_value_replay True --use_reward_prediction False --max_time_step 10000000 --checkpoint_dir /tmp/Pong_ftf_checkpoints --log_file /tmp/Pong_ftf_log
tensorboard --logdir=/tmp/Pong_ftf_log

python3 main.py --env_type gym --env_name BreakoutNoFrameskip-v4 --use_pixel_change False --use_value_replay False --use_reward_prediction False --max_time_step 10000000 --checkpoint_dir /tmp/Breakout_fff_checkpoints --log_file /tmp/Breakout_fff_log
tensorboard --logdir=/tmp/Breakout_fff_log
python3 main.py --env_type gym --env_name BreakoutNoFrameskip-v4 --use_pixel_change True --use_value_replay False --use_reward_prediction False --max_time_step 10000000 --checkpoint_dir /tmp/Breakout_tff_checkpoints --log_file /tmp/Breakout_tff_log
tensorboard --logdir=/tmp/Breakout_tff_log

python3 main.py --env_type gym --env_name BeamRiderNoFrameskip-v4 --use_pixel_change False --use_value_replay False --use_reward_prediction False --max_time_step 10000000 --checkpoint_dir /tmp/BeamRider_fff_checkpoints --log_file /tmp/BeamRider_fff_log
tensorboard --logdir=/tmp/BeamRider_fff_log
python3 main.py --env_type gym --env_name BeamRiderNoFrameskip-v4 --use_pixel_change True --use_value_replay False --use_reward_prediction False --max_time_step 10000000 --checkpoint_dir /tmp/BeamRider_tff_checkpoints --log_file /tmp/BeamRider_tff_log
tensorboard --logdir=/tmp/BeamRider_tff_log


# ======================================================================================


python3 main.py --env_type gym --env_name QbertNoFrameskip-v4 --use_pixel_change False --use_value_replay False --use_reward_prediction False --max_time_step 10000000 --checkpoint_dir /tmp/Qbert_fff_checkpoints --log_file /tmp/Qbert_fff_log
tensorboard --logdir=/tmp/Qbert_fff_log
python3 main.py --env_type gym --env_name QbertNoFrameskip-v4 --use_pixel_change True --use_value_replay False --use_reward_prediction False --max_time_step 10000000 --checkpoint_dir /tmp/Qbert_tff_checkpoints --log_file /tmp/Qbert_tff_log
tensorboard --logdir=/tmp/Qbert_tff_log

python3 main.py --env_type gym --env_name SpaceInvadersNoFrameskip-v4 --use_pixel_change False --use_value_replay False --use_reward_prediction False --max_time_step 10000000 --checkpoint_dir /tmp/SpaceInvaders_fff_checkpoints --log_file /tmp/SpaceInvaders_fff_log
tensorboard --logdir=gs://tmpp/SpaceInvaders_fff_log

python3 main.py --env_type gym --env_name AlienNoFrameskip-v4 --use_pixel_change True --use_value_replay True --use_reward_prediction True --checkpoint_dir /tmp/{}_checkpoints --frame_save_dir /tmp/{}_frames

python3 main.py --env_type gym --env_name AlienNoFrameskip-v4 --use_pixel_change True --use_value_replay True --use_reward_prediction True --checkpoint_dir /tmp/{}_checkpoints --frame_save_dir /tmp/{}_frames


python3 main.py --env_type gym --env_name PongNoFrameskip-v4 --use_pixel_change True --use_value_replay True --use_reward_prediction True --max_time_step 10000000 --checkpoint_dir /tmp/Pong_ttt_checkpoints_001 --log_file /tmp/Pong_ttt_log_001

