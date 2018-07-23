import logging

import tensorflow as tf
from unitytrainers.models import LearningModel

logger = logging.getLogger("unityagents")


class MADQNModel(LearningModel):
    def __init__(self, brain, lr=1e-4, h_size=128, epsilon_start=1, epsilon_end = 0.1,
                 epsilon_decay_steps=1e5, tau=0.0001, max_step=5e6,
                 normalize=False, use_recurrent=False, num_layers=2, m_size=None,
                 frozen=False, update_frozen_freq=None):
        """
        Takes a Unity environment and model-specific hyper-parameters and returns the
        appropriate MADQN agent model for the environment.
        :param brain: BrainInfo used to generate specific network graph.
        :param lr: Learning rate.
        :param h_size: Size of hidden layers
        :param epsilon: Value for policy-divergence threshold.
        :param beta: Strength of entropy regularization.
        :return: a sub-class of PPOAgent tailored to the environment.
        :param max_step: Total number of training steps.
        :param normalize: Whether to normalize vector observation input.
        :param use_recurrent: Whether to use an LSTM layer in the network.
        :param num_layers Number of hidden layers between encoded input and policy & value layers
        :param m_size: Size of brain memory.
        """
        LearningModel.__init__(self, m_size, normalize, use_recurrent, brain)
        if num_layers < 1:
            num_layers = 1
        self.last_reward, self.new_reward, self.update_reward = self.create_reward_encoder()
        if brain.vector_action_space_type == "continuous":
            print("DQN only supports discrete action space")
        else:
            self.create_madqn_model(h_size, num_layers, epsilon_start, epsilon_end, epsilon_decay_steps, frozen, update_frozen_freq)
        if not frozen:
            self.create_dqn_optimizer(lr, max_step)

    @staticmethod
    def create_reward_encoder():
        """Creates TF ops to track and increment recent average cumulative reward."""
        last_reward = tf.Variable(0, name="last_reward", trainable=False, dtype=tf.float32)
        new_reward = tf.placeholder(shape=[], dtype=tf.float32, name='new_reward')
        update_reward = tf.assign(last_reward, new_reward)
        return last_reward, new_reward, update_reward

    def create_dqn_optimizer(self, lr, max_step):
        """
        Creates training-specific Tensorflow ops for DQN model update.
        :param lr: Learning rate
        :param max_step: Total number of training steps
        """
        self.loss = tf.reduce_mean(tf.squared_difference(self.action_value, self.targets))
        self.learning_rate = tf.train.polynomial_decay(lr, self.global_step, max_step, 1e-10, power=1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update_batch = optimizer.minimize(self.loss)
