import tensorflow as tf
import numpy as np

import utils


class ModelBasedPolicy(object):

    def __init__(self,
                 env,
                 init_dataset,
                 horizon=15,
                 num_random_action_selection=4096,
                 nn_layers=1,
                 cem=False):
        self._cost_fn = env.cost_fn
        self._state_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._action_space_low = env.action_space.low
        self._action_space_high = env.action_space.high
        self._init_dataset = init_dataset
        self._horizon = horizon
        self._num_random_action_selection = num_random_action_selection
        self._nn_layers = nn_layers
        self._learning_rate = 1e-3

        self._cem = cem

        self._sess, self._state_ph, self._action_ph, self._next_state_ph,\
            self._next_state_pred, self._loss, self._optimizer, self._best_action = self._setup_graph()

    def _setup_placeholders(self):
        """
            Creates the placeholders used for training, prediction, and action selection

            returns:
                state_ph: current state
                action_ph: current_action
                next_state_ph: next state

            implementation details:
                (a) the placeholders should have 2 dimensions,
                    in which the 1st dimension is variable length (i.e., None)
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        ### BEGIN Solution
        state_ph = tf.placeholder(shape=[None, self._state_dim], name="state", dtype=tf.float32)

        action_ph = tf.placeholder(shape=[None, self._action_dim], name="action", dtype=tf.float32)

        next_state_ph = tf.placeholder(shape=[None, self._state_dim], name="next_state", dtype=tf.float32)
        ### END Solution

        return state_ph, action_ph, next_state_ph

    def _dynamics_func(self, state, action, reuse):
        """
            Takes as input a state and action, and predicts the next state

            returns:
                next_state_pred: predicted next state

            implementation details (in order):
                (a) Normalize both the state and action by using the statistics of self._init_dataset and
                    the utils.normalize function
                (b) Concatenate the normalized state and action
                (c) Pass the concatenated, normalized state-action tensor through a neural network with
                    self._nn_layers number of layers using the function utils.build_mlp. The resulting output
                    is the normalized predicted difference between the next state and the current state
                (d) Unnormalize the delta state prediction, and add it to the current state in order to produce
                    the predicted next state

        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        ### BEGIN Solution
        # Normalize state, action
        state_normalized = utils.normalize(state,
                                self._init_dataset.state_mean,
                                self._init_dataset.state_std)
        action_normalized = utils.normalize(action,
                                self._init_dataset.action_mean,
                                self._init_dataset.action_std)

        # concat state, action and use mlp to predict delta
        state_action_normalized = tf.concat([state_normalized, action_normalized], 1)
        delta_normalized = utils.build_mlp(input_layer=state_action_normalized,
                                    output_dim=self._state_dim,
                                    scope="dynamics",
                                    n_layers=self._nn_layers,
                                    reuse=reuse)
        delta = utils.unnormalize(delta_normalized,
                                self._init_dataset.delta_state_mean,
                                self._init_dataset.delta_state_std)

        # produce the predicted next state
        next_state_pred = state + delta
        ### END Solution

        return next_state_pred

    def _setup_training(self, state_ph, next_state_ph, next_state_pred):
        """
            Takes as input the current state, next state, and predicted next state, and returns
            the loss and optimizer for training the dynamics model

            returns:
                loss: Scalar loss tensor
                optimizer: Operation used to perform gradient descent

            implementation details (in order):
                (a) Compute both the actual state difference and the predicted state difference
                (b) Normalize both of these state differences by using the statistics of self._init_dataset and
                    the utils.normalize function
                (c) The loss function is the mean-squared-error between the normalized state difference and
                    normalized predicted state difference
                (d) Create the optimizer by minimizing the loss using the Adam optimizer with self._learning_rate

        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        ### BEGIN Solution
        # compuate actual state delta and pred state delta, normalize them
        state_delta_actual = next_state_ph - state_ph
        state_delta_pred = next_state_pred - state_ph

        state_delta_actual_normalized = utils.normalize(state_delta_actual,
                                    self._init_dataset.delta_state_mean,
                                    self._init_dataset.delta_state_std)
        state_delta_pred_normalized = utils.normalize(state_delta_pred,
                                    self._init_dataset.delta_state_mean,
                                    self._init_dataset.delta_state_std)

        # create loss and optimizer
        loss = tf.losses.mean_squared_error(state_delta_actual_normalized, state_delta_pred_normalized)
        optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)
        ### END Solution

        return loss, optimizer

    def _setup_action_selection(self, state_ph):
        """
            Computes the best action from the current state by using randomly sampled action sequences
            to predict future states, evaluating these predictions according to a cost function,
            selecting the action sequence with the lowest cost, and returning the first action in that sequence

            returns:
                best_action: the action that minimizes the cost function (tensor with shape [self._action_dim])

            implementation details (in order):
                (a) We will assume state_ph has a batch size of 1 whenever action selection is performed
                (b) Randomly sample uniformly self._num_random_action_selection number of action sequences,
                    each of length self._horizon
                (c) Starting from the input state, unroll each action sequence using your neural network
                    dynamics model
                (d) While unrolling the action sequences, keep track of the cost of each action sequence
                    using self._cost_fn
                (e) Find the action sequence with the lowest cost, and return the first action in that sequence

            Hints:
                (i) self._cost_fn takes three arguments: states, actions, and next states. These arguments are
                    2-dimensional tensors, where the 1st dimension is the batch size and the 2nd dimension is the
                    state or action size
                (ii) You should call self._dynamics_func and self._cost_fn a total of self._horizon times
                (iii) Use tf.random_uniform(...) to generate the random action sequences

        """
        ### PROBLEM 2
        ### YOUR CODE HERE
        # raise NotImplementedError
        ### BEGIN Solution
        if not self._cem:
            curr_state = tf.tile(state_ph, [self._num_random_action_selection, 1])
            action_random_sample = tf.random_uniform(shape=[self._horizon,
                                                            self._num_random_action_selection,
                                                            self._action_dim],
                                                    minval=self._action_space_low,
                                                    maxval=self._action_space_high,
                                                    dtype=tf.float32,
                                                    name="action_random_sample")
            cost = 0

            for i in range(self._horizon):
                next_state = self._dynamics_func(curr_state, action_random_sample[i], reuse=True)
                cost = cost + self._cost_fn(curr_state, action_random_sample[i], next_state)
                curr_state = next_state
                # action_random_sample = tf.random_uniform(shape=[self._num_random_action_selection, self._action_dim],
                #                                 minval=self._action_space_low,
                #                                 maxval=self._action_space_high,
                #                                 dtype=tf.float32,
                #                                 name="action_random_sample")
                # if i == 0:
                #     action_random_sample_t0 = action_random_sample
                #     cost = self._cost_fn(curr_state, action_random_sample, next_state)
                # else:
                #     cost = cost + self._cost_fn(curr_state, action_random_sample, next_state)

            best_action = action_random_sample[0][tf.argmin(cost)]
        else:
            ### Assume no covariance
            ### can be extended to only nearby cov or full cov
            mean, std = np.zeros(self._horizon), np.ones(self._horizon)
            num_random_action_selection = self._num_random_action_selection
            num_elite = int(0.10 * num_random_action_selection)
            num_cem_iter = 4

            for j in range(num_cem_iter):
                curr_state = tf.tile(state_ph, [self._num_random_action_selection, 1])
                action_random_sample = tf.convert_to_tensor(
                    [tf.clip_by_value(
                        tf.random_normal(
                            shape = [num_random_action_selection, self._action_dim],
                            mean = mean[ix],
                            stddev = std[ix],
                            dtype = tf.float32,
                            name = "action_random_sample_raw"
                            ),
                        clip_value_min = self._action_space_low,
                        clip_value_max = self._action_space_high,
                        name = "action_random_sample"
                    ) for ix in range(self._horizon)])
                cost = 0

                for i in range(self._horizon):
                    next_state = self._dynamics_func(curr_state, action_random_sample[i], reuse=True)
                    cost += self._cost_fn(curr_state, action_random_sample[i], next_state)
                    curr_state = next_state

                pos_elite = tf.nn.top_k(-cost, k = num_elite, sorted = True)
                # print(action_random_sample)
                # print(action_random_sample)
                action_elite = tf.gather(action_random_sample, pos_elite.indices, axis = 1)
                # action_elite = tf.gather(noodle, tf.range(chop_indices[list_idx,0], chop_indices[list_idx,1]))
                # print(action_elite)

                mean = tf.reduce_mean(action_elite, axis = 0)
                std = tf.sqrt(tf.reduce_mean(tf.square(action_elite-mean), axis = 0))
                # print(mean)

            best_action = mean[0]
        ### END Solution

        return best_action

    def _setup_graph(self):
        """
        Sets up the tensorflow computation graph for training, prediction, and action selection

        The variables returned will be set as class attributes (see __init__)
        """
        sess = tf.Session()

        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        ### BEGIN Solution
        # steup placeholdrs
        state_ph, action_ph, next_state_ph = self._setup_placeholders()

        # predict next state
        next_state_pred = self._dynamics_func(state_ph, action_ph, reuse=False)

        # setup training loss and opt
        loss, optimizer = self._setup_training(state_ph, next_state_ph, next_state_pred)
        ### END Solution

        ### PROBLEM 2
        ### YOUR CODE HERE
        ### BEGIN Solution
        # choose the best action
        best_action =  self._setup_action_selection(state_ph)
        ### END Solution

        sess.run(tf.global_variables_initializer())

        return sess, state_ph, action_ph, next_state_ph, \
                next_state_pred, loss, optimizer, best_action

    def train_step(self, states, actions, next_states):
        """
        Performs one step of gradient descent

        returns:
            loss: the loss from performing gradient descent
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        ### BEGIN Solution
        # optimizer take one step and record the loss
        _, loss = self._sess.run([self._optimizer, self._loss],
                            feed_dict={self._state_ph: states,
                                        self._action_ph: actions,
                                        self._next_state_ph: next_states})
        ### END Solution

        return loss

    def predict(self, state, action):
        """
        Predicts the next state given the current state and action

        returns:
            next_state_pred: predicted next state

        implementation detils:
            (i) The state and action arguments are 1-dimensional vectors (NO batch dimension)
        """
        assert np.shape(state) == (self._state_dim,)
        assert np.shape(action) == (self._action_dim,)


        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        ### BEGIN Solution
        # predict next state
        next_state_pred = self._sess.run(self._next_state_pred,
                                feed_dict={self._state_ph: np.reshape(state, (1, state.shape[0])),
                                            self._action_ph: np.reshape(action, (1, action.shape[0]))})
        next_state_pred = np.reshape(next_state_pred, (next_state_pred.shape[1], ))
        ### END Solution

        assert np.shape(next_state_pred) == (self._state_dim,)
        return next_state_pred

    def get_action(self, state):
        """
        Computes the action that minimizes the cost function given the current state

        returns:
            best_action: the best action
        """
        assert np.shape(state) == (self._state_dim,)

        ### PROBLEM 2
        ### YOUR CODE HERE
        # raise NotImplementedError
        ### BEGIN Solution
        # get the best action using random shooting method
        best_action = self._sess.run(self._best_action,
                                feed_dict={self._state_ph: np.reshape(state, (1, state.shape[0]))})
        best_action = np.reshape(best_action, (best_action.shape[0], ))
        ### END Solution

        assert np.shape(best_action) == (self._action_dim,)
        return best_action
