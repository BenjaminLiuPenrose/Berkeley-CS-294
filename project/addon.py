def meta_run(self):
    # flags.pixel_change_lambda, flags.value_replay_lambda, flags.reward_prediction_lambda
    weight_meta = [1,1,1]
    step_0 = 10000
    step_1 = 100000
    num_iter_meta = int(flags.max_time_step/step_1)
    for iter in range(num_iter_meta):
        self.total_reward = []
        self.run_train(True,False,False,weight_meta[0]/1000,weight_meta[1],weight_meta[2],False,True,step_0)
        self.run_train(False,True,False,weight_meta[0]/1000,weight_meta[1],weight_meta[2],False,True,step_0)
        self.run_train(False,False,True,weight_meta[0]/1000,weight_meta[1],weight_meta[2],False,True,step_0)
        max_idx = self.total_reward.index(max(self.total_reward))[0]
        min_idx = self.total_reward.index(min(self.total_reward))[0]
        weight_meta[max_idx] *= 1.05
        weight_meta[min_idx] *= 0.95
        self.run_train(True,True,True,weight_meta[0]/100,weight_meta[1],weight_meta[2],True,True,step_1)


def run_train(self,pc,vr,rp,step_size,pc_w,vr_w,rp_w,save,load_cp,step):
    self.current_reward = 0
    self.mod_run(pc,vr,rp,pc_w,vr_w,rp_w,save,load_cp,step)
    for (i, t) in enumerate(self.train_threads):
        if i != 0:
            t.join()
    self.total_reward.append(self.current_reward)

def mod_run(self,pc,vr,rp,step_size,pc_w,vr_w,rp_w,save,load_cp,step):
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
        with open(wall_t_fname, 'r') as f:
        self.wall_t = float(f.read())
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

    signal.signal(signal.SIGINT, self.signal_handler)

    # set start time
    self.start_time = time.time() - self.wall_t

    for t in self.train_threads:
        t.start()


    signal.pause()

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
            trainer.stop()
            if parallel_index == 0 and save:
                self.mod_save()
            break
        if parallel_index == 0 and self.global_t > self.next_save_steps and save:
            # Save checkpoint
            self.save()
        # logging rewards
        diff_global_t,total_reward = trainer.process(self.sess,
                                        self.global_t,
                                        self.summary_writer,
                                        self.summary_op,
                                        self.score_input)
        self.global_t += diff_global_t
        self.current_reward += total_reward



def mod_save(self):
    """ Save checkpoint.
    Called from therad-0.
    """
    # logging
    self.stop_requested = True

    # Save
    if not os.path.exists(flags.checkpoint_dir):
      os.mkdir(flags.checkpoint_dir)

    # # Write wall time
    # wall_t = time.time() - self.start_time
    # wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)
    # with open(wall_t_fname, 'w') as f:
    #   f.write(str(wall_t))

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
