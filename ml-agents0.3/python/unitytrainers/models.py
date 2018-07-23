import logging

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as c_layers

logger = logging.getLogger("unityagents")


class LearningModel(object):
    def __init__(self, m_size, normalize, use_recurrent, brain):
        self.brain = brain
        self.vector_in = None
        self.normalize = False
        self.use_recurrent = False
        self.global_step, self.increment_step = self.create_global_steps()
        self.visual_in = []
        self.batch_size = tf.placeholder(shape=None, dtype=tf.int32, name='batch_size')
        self.sequence_length = tf.placeholder(shape=None, dtype=tf.int32, name='sequence_length')
        self.m_size = m_size
        self.normalize = normalize
        self.use_recurrent = use_recurrent
        self.a_size = brain.vector_action_space_size

    @staticmethod
    def create_global_steps():
        """Creates TF ops to track and increment global training step."""
        global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)
        increment_step = tf.assign(global_step, tf.add(global_step, 1))
        return global_step, increment_step

    @staticmethod
    def swish(input_activation):
        """Swish activation function. For more info: https://arxiv.org/abs/1710.05941"""
        return tf.multiply(input_activation, tf.nn.sigmoid(input_activation))

    @staticmethod
    def create_visual_input(o_size_h, o_size_w, bw, name):
        if bw:
            c_channels = 1
        else:
            c_channels = 3

        visual_in = tf.placeholder(shape=[None, o_size_h, o_size_w, c_channels], dtype=tf.float32, name=name)
        return visual_in

    def create_vector_input(self, s_size):
        if self.brain.vector_observation_space_type == "continuous":
            self.vector_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32, name='vector_observation')
            if self.normalize:
                self.running_mean = tf.get_variable("running_mean", [s_size], trainable=False, dtype=tf.float32,
                                                    initializer=tf.zeros_initializer())
                self.running_variance = tf.get_variable("running_variance", [s_size], trainable=False, dtype=tf.float32,
                                                        initializer=tf.ones_initializer())
                self.new_mean = tf.placeholder(shape=[s_size], dtype=tf.float32, name='new_mean')
                self.new_variance = tf.placeholder(shape=[s_size], dtype=tf.float32, name='new_variance')
                self.update_mean = tf.assign(self.running_mean, self.new_mean)
                self.update_variance = tf.assign(self.running_variance, self.new_variance)

                self.normalized_state = tf.clip_by_value((self.vector_in - self.running_mean) / tf.sqrt(
                    self.running_variance / (tf.cast(self.global_step, tf.float32) + 1)), -5, 5,
                                                         name="normalized_state")
            else:
                self.normalized_state = self.vector_in

        else:
            self.vector_in = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='vector_observation')

    def create_continuous_state_encoder(self, h_size, activation, num_layers):
        """
        Builds a set of hidden state encoders.
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :return: List of hidden layer tensors.
        """
        hidden = self.normalized_state
        for j in range(num_layers):
            hidden = tf.layers.dense(hidden, h_size, activation=activation,
                                     kernel_initializer=c_layers.variance_scaling_initializer(1.0))
        return hidden

    def create_continuous_state_action_encoder(self, h_size, activation, num_layers):
        """
        Builds a set of hidden state encoders.
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :return: List of hidden layer tensors.
        """
        # Actions taken by other agents
        self.other_actions = tf.placeholder(shape=[None, self.n_brain - 1], dtype=tf.int64, name='other_actions')
        self.other_actions_one_hot = c_layers.one_hot_encoding(self.other_actions, self.a_size)
        self.other_actions_one_hot = tf.reshape(self.other_actions_one_hot, [-1,self.a_size*(self.n_brain - 1)])
        self.other_actions_one_hot = tf.cast(self.other_actions_one_hot, tf.float32)
        hidden = tf.concat([self.normalized_state, self.other_actions_one_hot], axis=1)
        for j in range(num_layers):
            hidden = tf.layers.dense(hidden, h_size, activation=activation,
                                     kernel_initializer=c_layers.variance_scaling_initializer(1.0))
        return hidden

    def create_visual_encoder(self, h_size, activation, num_layers):
        """
        Builds a set of visual (CNN) encoders.
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :return: List of hidden layer tensors.
        """
        conv1 = tf.layers.conv2d(self.visual_in[-1], 16, kernel_size=[8, 8], strides=[4, 4],
                                 activation=tf.nn.elu)
        conv2 = tf.layers.conv2d(conv1, 32, kernel_size=[4, 4], strides=[2, 2],
                                 activation=tf.nn.elu)
        hidden = c_layers.flatten(conv2)

        for j in range(num_layers):
            hidden = tf.layers.dense(hidden, h_size, use_bias=False, activation=activation)
        return hidden

    def create_discrete_state_encoder(self, s_size, h_size, activation, num_layers):
        """
        Builds a set of hidden state encoders from discrete state input.
        :param s_size: state input size (discrete).
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :return: List of hidden layer tensors.
        """
        vector_in = tf.reshape(self.vector_in, [-1])
        state_onehot = c_layers.one_hot_encoding(vector_in, s_size)
        hidden = state_onehot
        for j in range(num_layers):
            hidden = tf.layers.dense(hidden, h_size, use_bias=False, activation=activation)
        return hidden

    def create_new_obs(self, num_streams, h_size, num_layers):
        brain = self.brain
        s_size = brain.vector_observation_space_size * brain.num_stacked_vector_observations
        if brain.vector_action_space_type == "continuous":
            activation_fn = tf.nn.tanh
        else:
            activation_fn = self.swish

        self.visual_in = []
        for i in range(brain.number_visual_observations):
            height_size, width_size = brain.camera_resolutions[i]['height'], brain.camera_resolutions[i]['width']
            bw = brain.camera_resolutions[i]['blackAndWhite']
            visual_input = self.create_visual_input(height_size, width_size, bw, name="visual_observation_" + str(i))
            self.visual_in.append(visual_input)
        self.create_vector_input(s_size)

        final_hiddens = []
        for i in range(num_streams):
            visual_encoders = []
            hidden_state, hidden_visual = None, None
            if brain.number_visual_observations > 0:
                for j in range(brain.number_visual_observations):
                    encoded_visual = self.create_visual_encoder(h_size, activation_fn, num_layers)
                    visual_encoders.append(encoded_visual)
                hidden_visual = tf.concat(visual_encoders, axis=1)
            if brain.vector_observation_space_size > 0:
                s_size = brain.vector_observation_space_size * brain.num_stacked_vector_observations
                if brain.vector_observation_space_type == "continuous":
                    hidden_state = self.create_continuous_state_encoder(h_size, activation_fn, num_layers)
                else:
                    hidden_state = self.create_discrete_state_encoder(s_size, h_size,
                                                                      activation_fn, num_layers)
            if hidden_state is not None and hidden_visual is not None:
                final_hidden = tf.concat([hidden_visual, hidden_state], axis=1)
            elif hidden_state is None and hidden_visual is not None:
                final_hidden = hidden_visual
            elif hidden_state is not None and hidden_visual is None:
                final_hidden = hidden_state
            else:
                raise Exception("No valid network configuration possible. "
                                "There are no states or observations in this brain")
            final_hiddens.append(final_hidden)
        return final_hiddens

    def create_new_action_obs(self, num_streams, h_size, num_layers, trainer):
        brain = self.brain
        s_size = brain.vector_observation_space_size * brain.num_stacked_vector_observations
        if brain.vector_action_space_type == "continuous":
            activation_fn = tf.nn.tanh
        else:
            activation_fn = self.swish

        self.visual_in = []
        for i in range(brain.number_visual_observations):
            height_size, width_size = brain.camera_resolutions[i]['height'], brain.camera_resolutions[i]['width']
            bw = brain.camera_resolutions[i]['blackAndWhite']
            visual_input = self.create_visual_input(height_size, width_size, bw, name="visual_observation_" + str(i))
            self.visual_in.append(visual_input)
        #self.create_vector_input(s_size)

        final_hiddens = []
        for i in range(num_streams):
            visual_encoders = []
            hidden_state, hidden_visual = None, None
            if brain.number_visual_observations > 0:
                for j in range(brain.number_visual_observations):
                    encoded_visual = self.create_visual_encoder(h_size, activation_fn, num_layers)
                    visual_encoders.append(encoded_visual)
                hidden_visual = tf.concat(visual_encoders, axis=1)
            if brain.vector_observation_space_size > 0:
                s_size = brain.vector_observation_space_size * brain.num_stacked_vector_observations
                if brain.vector_observation_space_type == "continuous":
                    hidden_state_action = self.create_continuous_state_action_encoder(h_size, activation_fn, num_layers)
                else:
                    hidden_state = self.create_discrete_state_encoder(s_size, h_size,
                                                                      activation_fn, num_layers)
            if hidden_state is not None and hidden_visual is not None:
                final_hidden = tf.concat([hidden_visual, hidden_state], axis=1)
            elif hidden_state is None and hidden_visual is not None:
                final_hidden = hidden_visual
            elif hidden_state_action is not None and hidden_visual is None:
                final_hidden = hidden_state_action
            else:
                raise Exception("No valid network configuration possible. "
                                "There are no states or observations in this brain")
            final_hiddens.append(final_hidden)
        return final_hiddens

    def create_recurrent_encoder(self, input_state, memory_in, name='lstm'):
        """
        Builds a recurrent encoder for either state or observations (LSTM).
        :param input_state: The input tensor to the LSTM cell.
        :param memory_in: The input memory to the LSTM cell.
        :param name: The scope of the LSTM cell.
        """
        s_size = input_state.get_shape().as_list()[1]
        m_size = memory_in.get_shape().as_list()[1]
        lstm_input_state = tf.reshape(input_state, shape=[-1, self.sequence_length, s_size])
        _half_point = int(m_size / 2)
        with tf.variable_scope(name):
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(_half_point)
            lstm_vector_in = tf.contrib.rnn.LSTMStateTuple(memory_in[:, :_half_point], memory_in[:, _half_point:])
            recurrent_state, lstm_state_out = tf.nn.dynamic_rnn(rnn_cell, lstm_input_state,
                                                                initial_state=lstm_vector_in,
                                                                time_major=False,
                                                                dtype=tf.float32)

        recurrent_state = tf.reshape(recurrent_state, shape=[-1, _half_point])
        return recurrent_state, tf.concat([lstm_state_out.c, lstm_state_out.h], axis=1)

    def create_dc_actor_critic(self, h_size, num_layers):
        num_streams = 1
        hidden_streams = self.create_new_obs(num_streams, h_size, num_layers)
        hidden = hidden_streams[0]

        if self.use_recurrent:
            tf.Variable(self.m_size, name="memory_size", trainable=False, dtype=tf.int32)
            self.prev_action = tf.placeholder(shape=[None], dtype=tf.int32, name='prev_action')
            self.prev_action_oh = c_layers.one_hot_encoding(self.prev_action, self.a_size)
            hidden = tf.concat([hidden, self.prev_action_oh], axis=1)

            self.memory_in = tf.placeholder(shape=[None, self.m_size], dtype=tf.float32, name='recurrent_in')
            hidden, self.memory_out = self.create_recurrent_encoder(hidden, self.memory_in)
            self.memory_out = tf.identity(self.memory_out, name='recurrent_out')

        self.policy = tf.layers.dense(hidden, self.a_size, activation=None, use_bias=False,
                                      kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01))

        self.all_probs = tf.nn.softmax(self.policy, name="action_probs")
        self.output = tf.multinomial(self.policy, 1)
        self.output = tf.identity(self.output, name="action")

        self.value = tf.layers.dense(hidden, 1, activation=None)
        self.value = tf.identity(self.value, name="value_estimate")
        self.entropy = -tf.reduce_sum(self.all_probs * tf.log(self.all_probs + 1e-10), axis=1)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.selected_actions = c_layers.one_hot_encoding(self.action_holder, self.a_size)

        self.all_old_probs = tf.placeholder(shape=[None, self.a_size], dtype=tf.float32, name='old_probabilities')

        # We reshape these tensors to [batch x 1] in order to be of the same rank as continuous control probabilities.
        self.probs = tf.expand_dims(tf.reduce_sum(self.all_probs * self.selected_actions, axis=1), 1)
        self.old_probs = tf.expand_dims(tf.reduce_sum(self.all_old_probs * self.selected_actions, axis=1), 1)

    def create_dc_ma_actor_critic(self, h_size, num_layers, n_brain, trainer):
        num_streams = 1
        self.n_brain = n_brain
        hidden_streams1 = self.create_new_obs(num_streams, h_size, num_layers)
        hidden1 = hidden_streams1[0]
        # hidden_streams2 = self.create_new_action_obs(num_streams, h_size, num_layers, trainer)
        # hidden2 = hidden_streams2[0]

        if self.use_recurrent:
            tf.Variable(self.m_size, name="memory_size", trainable=False, dtype=tf.int32)
            self.prev_action = tf.placeholder(shape=[None], dtype=tf.int32, name='prev_action')
            self.prev_action_oh = c_layers.one_hot_encoding(self.prev_action, self.a_size)
            hidden = tf.concat([hidden, self.prev_action_oh], axis=1)

            self.memory_in = tf.placeholder(shape=[None, self.m_size], dtype=tf.float32, name='recurrent_in')
            hidden, self.memory_out = self.create_recurrent_encoder(hidden, self.memory_in)
            self.memory_out = tf.identity(self.memory_out, name='recurrent_out')

        self.policy = tf.layers.dense(hidden1, self.a_size, activation=None, use_bias=False,
                                      kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01))

        #hidden2 = tf.layers.dense(hidden2, h_size, activation=self.swish)
        self.all_probs = tf.nn.softmax(self.policy, name="action_probs")
        self.output = tf.multinomial(self.policy, 1)
        self.output = tf.identity(self.output, name="action")

        self.other_actions = tf.placeholder(shape=[None, self.n_brain - 1], dtype=tf.int64, name='other_actions')
        self.other_actions_one_hot = c_layers.one_hot_encoding(self.other_actions, self.a_size)
        self.other_actions_one_hot = tf.reshape(self.other_actions_one_hot, [-1,self.a_size*(self.n_brain - 1)])
        self.other_actions_one_hot = tf.cast(self.other_actions_one_hot, tf.float32)
        hidden2 = tf.concat([hidden1, self.other_actions_one_hot], axis=1)
        self.value = tf.layers.dense(hidden2, 1, activation=None)
        self.value = tf.identity(self.value, name="value_estimate")

        self.entropy = -tf.reduce_sum(self.all_probs * tf.log(self.all_probs + 1e-10), axis=1)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.selected_actions = c_layers.one_hot_encoding(self.action_holder, self.a_size)

        self.all_old_probs = tf.placeholder(shape=[None, self.a_size], dtype=tf.float32, name='old_probabilities')

        # We reshape these tensors to [batch x 1] in order to be of the same rank as continuous control probabilities.
        self.probs = tf.expand_dims(tf.reduce_sum(self.all_probs * self.selected_actions, axis=1), 1)
        self.old_probs = tf.expand_dims(tf.reduce_sum(self.all_old_probs * self.selected_actions, axis=1), 1)

    def create_dc_coma_actor_critic(self, h_size, num_layers, n_brain, trainer):
        num_streams = 1
        self.n_brain = n_brain
        hidden_streams1 = self.create_new_obs(num_streams, h_size, num_layers)
        hidden1 = hidden_streams1[0]
        # hidden_streams2 = self.create_new_action_obs(num_streams, h_size, num_layers, trainer)
        # hidden2 = hidden_streams2[0]

        if self.use_recurrent:
            tf.Variable(self.m_size, name="memory_size", trainable=False, dtype=tf.int32)
            self.prev_action = tf.placeholder(shape=[None], dtype=tf.int32, name='prev_action')
            self.prev_action_oh = c_layers.one_hot_encoding(self.prev_action, self.a_size)
            hidden = tf.concat([hidden, self.prev_action_oh], axis=1)

            self.memory_in = tf.placeholder(shape=[None, self.m_size], dtype=tf.float32, name='recurrent_in')
            hidden, self.memory_out = self.create_recurrent_encoder(hidden, self.memory_in)
            self.memory_out = tf.identity(self.memory_out, name='recurrent_out')

        self.policy = tf.layers.dense(hidden1, self.a_size, activation=None, use_bias=False,
                                      kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01))

        self.all_probs = tf.nn.softmax(self.policy, name="action_probs")
        self.output = tf.multinomial(self.policy, 1)
        self.output = tf.identity(self.output, name="action")

        #self.other_actions = tf.placeholder(shape=[None, self.n_brain - 1], dtype=tf.int64, name='other_actions')
        self.agent_action = tf.placeholder(shape=[None, 1], dtype=tf.int64, name="agent_action")
        self.agent_action_one_hot = tf.one_hot(self.agent_action, self.a_size, dtype=tf.float32)
        self.other_actions = tf.placeholder(shape=[None, self.n_brain - 1], dtype=tf.int64, name='other_actions')
        self.other_actions_one_hot = c_layers.one_hot_encoding(self.other_actions, self.a_size)
        self.other_actions_one_hot = tf.reshape(self.other_actions_one_hot, [-1,self.a_size*(self.n_brain - 1)])
        self.other_actions_one_hot = tf.cast(self.other_actions_one_hot, tf.float32)
        hidden2 = tf.concat([hidden1, self.other_actions_one_hot], axis=1)
        self.values = tf.layers.dense(hidden2, self.a_size, activation=None)
        self.value = tf.reduce_sum(tf.multiply(self.values, tf.reshape(self.agent_action_one_hot,[-1, self.a_size])), axis=1)
        self.value = tf.identity(self.value, name="value_estimate")
        self.entropy = -tf.reduce_sum(self.all_probs * tf.log(self.all_probs + 1e-10), axis=1)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.selected_actions = c_layers.one_hot_encoding(self.action_holder, self.a_size)

        self.all_old_probs = tf.placeholder(shape=[None, self.a_size], dtype=tf.float32, name='old_probabilities')

        # We reshape these tensors to [batch x 1] in order to be of the same rank as continuous control probabilities.
        self.probs = tf.expand_dims(tf.reduce_sum(self.all_probs * self.selected_actions, axis=1), 1)
        self.old_probs = tf.expand_dims(tf.reduce_sum(self.all_old_probs * self.selected_actions, axis=1), 1)

    def create_cc_actor_critic(self, h_size, num_layers):
        num_streams = 2
        hidden_streams = self.create_new_obs(num_streams, h_size, num_layers)

        if self.use_recurrent:
            tf.Variable(self.m_size, name="memory_size", trainable=False, dtype=tf.int32)
            self.memory_in = tf.placeholder(shape=[None, self.m_size], dtype=tf.float32, name='recurrent_in')
            _half_point = int(self.m_size / 2)
            hidden_policy, memory_policy_out = self.create_recurrent_encoder(
                hidden_streams[0], self.memory_in[:, :_half_point], name='lstm_policy')

            hidden_value, memory_value_out = self.create_recurrent_encoder(
                hidden_streams[1], self.memory_in[:, _half_point:], name='lstm_value')
            self.memory_out = tf.concat([memory_policy_out, memory_value_out], axis=1, name='recurrent_out')
        else:
            hidden_policy = hidden_streams[0]
            hidden_value = hidden_streams[1]

        self.mu = tf.layers.dense(hidden_policy, self.a_size, activation=None, use_bias=False,
                                  kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01))

        self.log_sigma_sq = tf.get_variable("log_sigma_squared", [self.a_size], dtype=tf.float32,
                                            initializer=tf.zeros_initializer())

        self.sigma_sq = tf.exp(self.log_sigma_sq)
        self.epsilon = tf.random_normal(tf.shape(self.mu), dtype=tf.float32)
        self.output = self.mu + tf.sqrt(self.sigma_sq) * self.epsilon
        self.output = tf.identity(self.output, name='action')
        a = tf.exp(-1 * tf.pow(tf.stop_gradient(self.output) - self.mu, 2) / (2 * self.sigma_sq))
        b = 1 / tf.sqrt(2 * self.sigma_sq * np.pi)
        self.all_probs = tf.multiply(a, b, name="action_probs")
        self.entropy = tf.reduce_mean(0.5 * tf.log(2 * np.pi * np.e * self.sigma_sq))
        self.value = tf.layers.dense(hidden_value, 1, activation=None)
        self.value = tf.identity(self.value, name="value_estimate")
        self.all_old_probs = tf.placeholder(shape=[None, self.a_size], dtype=tf.float32,
                                            name='old_probabilities')
        # We keep these tensors the same name, but use new nodes to keep code parallelism with discrete control.
        self.probs = tf.identity(self.all_probs)
        self.old_probs = tf.identity(self.all_old_probs)

    def create_dqn_model(self, h_size, num_layers, epsilon_start, epsilon_end, epsilon_decay_steps):
        """
        Creates Discrete Control Q learning model.
        :param brain: State-space size
        :param h_size: Hidden layer size
        :param max_step: Total number of training steps
        :param normalize: Normalize observations
        :num_layers: Number of hidden layers in the value function approximator
        """
        num_streams = 1
        hidden_streams = self.create_new_obs(num_streams, h_size, num_layers)
        hidden = hidden_streams[0]

        if self.use_recurrent:
            tf.Variable(self.m_size, name="memory_size", trainable=False, dtype=tf.int32)
            self.prev_action = tf.placeholder(shape=[None], dtype=tf.int32, name='prev_action')
            self.prev_action_oh = c_layers.one_hot_encoding(self.prev_action, self.a_size)
            hidden = tf.concat([hidden, self.prev_action_oh], axis=1)

            self.memory_in = tf.placeholder(shape=[None, self.m_size], dtype=tf.float32, name='recurrent_in')
            hidden, self.memory_out = self.create_recurrent_encoder(hidden, self.memory_in)
            self.memory_out = tf.identity(self.memory_out, name='recurrent_out')

        # Value Function predictions
        self.predictions = tf.layers.dense(hidden, self.a_size, activation=None, use_bias=True,
                                     kernel_initializer=c_layers.variance_scaling_initializer(factor=1.0))
        self.predictions = tf.identity(self.predictions, name="value_estimate")
        self.output = tf.argmax(self.predictions,1)
        self.epsilon = tf.train.polynomial_decay(epsilon_start, self.global_step,
                                                epsilon_decay_steps, epsilon_end,
                                                power=1.0)
        # self.explore = tf.random_uniform([2], 0, 1) < self.epsilon
        # self.chosen_action = tf.cond(self.explore, lambda: self.random_action, lambda: self.output)
        self.output_onehot = tf.one_hot(self.output, self.a_size, dtype=tf.float32)
        self.prob = tf.ones([self.a_size])*self.epsilon/self.a_size + tf.multiply(self.output_onehot,tf.ones([self.a_size])*(1-self.epsilon))
        self.prob = tf.identity(self.prob, name="action_probs")
        self.chosen_action = tf.multinomial(tf.log(self.prob), 1)
        self.output = tf.reshape(self.output,[tf.shape(self.vector_in)[0],1])
        self.output = tf.identity(self.output, name='action')

        # Value Function update
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="targets")
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")


        self.actions_onehot = tf.one_hot(self.actions, self.a_size, dtype=tf.float32)
        self.action_value = tf.reduce_sum(tf.multiply(self.predictions, self.actions_onehot), axis=1)

    def create_madqn_model(self, h_size, num_layers, epsilon_start, epsilon_end, epsilon_decay_steps, frozen, update_frozen_freq):
        """
        Creates Discrete Control Q learning model.
        :param brain: State-space size
        :param h_size: Hidden layer size
        :param max_step: Total number of training steps
        :param normalize: Normalize observations
        :num_layers: Number of hidden layers in the value function approximator
        """
        if not frozen: # Non-frozen brains are continously being updated
            num_streams = 1
            hidden_streams = self.create_new_obs(num_streams, h_size, num_layers)
            hidden = hidden_streams[0]
            if self.use_recurrent:
                tf.Variable(self.m_size, name="memory_size", trainable=False, dtype=tf.int32)
                self.prev_action = tf.placeholder(shape=[None], dtype=tf.int32, name='prev_action')
                self.prev_action_oh = c_layers.one_hot_encoding(self.prev_action, self.a_size)
                hidden = tf.concat([hidden, self.prev_action_oh], axis=1)

                self.memory_in = tf.placeholder(shape=[None, self.m_size], dtype=tf.float32, name='recurrent_in')
                hidden, self.memory_out = self.create_recurrent_encoder(hidden, self.memory_in)
                self.memory_out = tf.identity(self.memory_out, name='recurrent_out')

            # Value Function predictions
            self.predictions = tf.layers.dense(hidden, self.a_size, activation=None, use_bias=True,
                                         kernel_initializer=c_layers.variance_scaling_initializer(factor=1.0))
            self.predictions = tf.identity(self.predictions, name="value_estimate")
            self.output = tf.argmax(self.predictions,1)

            # Epsilon decay is restarted every time the frozen policy is updated
            #self.step =  self.global_step - self.n_updates*update_frozen_freq
            self.epsilon = tf.train.polynomial_decay(epsilon_start, self.global_step,
                                                    epsilon_decay_steps, epsilon_end,
                                                    power=1.0)

            self.output_onehot = tf.one_hot(self.output, self.a_size, dtype=tf.float32)
            self.prob = tf.ones([self.a_size])*self.epsilon/self.a_size + tf.multiply(self.output_onehot,tf.ones([self.a_size])*(1-self.epsilon))
            self.prob = tf.identity(self.prob, name="action_probs")
            self.chosen_action = tf.multinomial(tf.log(self.prob), 1)
            self.output = tf.reshape(self.output,[tf.shape(self.vector_in)[0],1])
            self.output = tf.identity(self.output, name="action")

            # Value Function update
            self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="targets")
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")


            self.actions_onehot = tf.one_hot(self.actions, self.a_size, dtype=tf.float32)
            self.action_value = tf.reduce_sum(tf.multiply(self.predictions, self.actions_onehot), axis=1)

        else: # Frozen brains are only updated periodically
            num_streams = 1
            hidden_streams = self.create_new_obs(num_streams, h_size, num_layers)
            hidden = hidden_streams[0]
            if self.use_recurrent:
                tf.Variable(self.m_size, name="memory_size", trainable=False, dtype=tf.int32)
                self.prev_action = tf.placeholder(shape=[None], dtype=tf.int32, name='prev_action')
                self.prev_action_oh = c_layers.one_hot_encoding(self.prev_action, self.a_size)
                hidden = tf.concat([hidden, self.prev_action_oh], axis=1)

                self.memory_in = tf.placeholder(shape=[None, self.m_size], dtype=tf.float32, name='recurrent_in')
                hidden, self.memory_out = self.create_recurrent_encoder(hidden, self.memory_in)
                self.memory_out = tf.identity(self.memory_out, name='recurrent_out')
            # Value Function predictions
            self.predictions = tf.layers.dense(hidden, self.a_size, activation=None, use_bias=True,
                                         kernel_initializer=c_layers.variance_scaling_initializer(factor=1.0))
            self.predictions = tf.identity(self.predictions, name="value_estimate")
            self.output = tf.argmax(self.predictions,1)
            self.output_onehot = tf.one_hot(self.output, self.a_size, dtype=tf.float32)
            self.epsilon = tf.train.polynomial_decay(epsilon_start, self.global_step,
                                                    epsilon_decay_steps, epsilon_end,
                                                    power=1.0)
            self.prob = tf.ones([self.a_size])*self.epsilon/self.a_size + tf.multiply(self.output_onehot,tf.ones([self.a_size])*(1-self.epsilon))
            self.prob = tf.identity(self.prob, name="action_probs")
            self.chosen_action = tf.multinomial(tf.log(self.prob), 1)
            self.output = tf.reshape(self.output,[tf.shape(self.vector_in)[0],1])
            self.output = tf.identity(self.output, name="action")
