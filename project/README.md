# ProjectUnreal
Project for Berkeley DRL

run `python main.py`
<<<<<<< HEAD
or run `python main.py --env_type gym --env_name [your_game]`


Update @ 11/16
game:
(PongNoFrameskip-v4, tmp/Pong_log/[experiemnt type])
(BreakoutNoFrameskip-v4, tmp/Breakout_log/[experiemnt type])
(BeamRiderNoFrameskip-v4, tmp/BeamRider_log/[experiemnt type])
--
(QbertNoFrameskip-v4, tmp/Qbert_log/[experiemnt type])
(SpaceInvadersNoFrameskip-v4, tmp/SpaceInvaders_log/[experiemnt type])
Seaquest


experimet: (pc, vr, rp)
(False, False, False),
(True, False, False),
(False, True, False),
(False, False, True),
(True, True, True)


zx:
experiment: (True, True, True), (False, False, True) for Pong Breakout Beamrider

be:
experiment: (False, False, False), (True, False, False), for Pong Breakout Beamrider

Update @ 11/26
图放一个中 - Pong, Breakout
2 trues 3 games
prepare plots and nn pics

zx:
all experiments for Seaquest (False, False, False), (False, False, True), (False, True, True), (True, True, True)
prepare nn arch pics

be:
experiment: (False, True, True) for Pong, Breakout, (False, True, False) for Seaquest
tanscript, ppt

Meet @ 11/30
图放一个中 - Pong, Breakout
add more aux tasks



sample cmd is in `run_all.sh`
atrai game list and ref rewards is `env_name_list.txt`

http://cs231n.stanford.edu/reports/2017/pdfs/610.pdf
other tasks mentioned before?
=======
or run `python3 main.py --env_type gym --env_name {}NoFrameskip-v4 --use_pixel_change True --use_value_replay True --use_reward_prediction True --max_time_step {}000000 --checkpoint_dir /tmp/{}_checkpoints --log_file /tmp/{}_log
tensorboard --logdir=/tmp/{}_log` where {} is name of Atari game
>>>>>>> 8c01deedc3a174afc0f2dbf07d305d4b223db5b6
