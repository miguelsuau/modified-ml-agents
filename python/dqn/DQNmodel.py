import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as c_layers
from tensorflow.python.tools import freeze_graph
from unityagents import UnityEnvironmentException


def create_agent_model(env, lr=1e-4, h_size=128, max_step=5e6, normalize=False, num_layers=2):
    """
    Takes a Unity environment and model-specific hyper-parameters and returns the
    appropriate PPO agent model for the environment.
    :param env: a Unity environment.
    :param lr: Learning rate.
    :param h_size: Size of hidden layers/
    :param epsilon: Value for policy-divergence threshold.
    :param beta: Strength of entropy regularization.
    :return: a sub-class of PPOAgent tailored to the environment.
    :param max_step: Total number of training steps.
    """
    if num_layers < 1: num_layers = 1

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    if brain.action_space_type == "continuous":
        return ContinuousControlModel(lr, brain, h_size, max_step, normalize, num_layers)
    if brain.action_space_type == "discrete":
        return DiscreteControlModel(lr, brain, h_size, max_step, normalize, num_layers)


def save_model(sess, saver, model_path="./", steps=0):
    """
    Saves current model to checkpoint folder.
    :param sess: Current Tensorflow session.
    :param model_path: Designated model path.
    :param steps: Current number of steps in training process.
    :param saver: Tensorflow saver for session.
    """
    last_checkpoint = model_path + '/model-' + str(steps) + '.cptk'
    saver.save(sess, last_checkpoint)
    tf.train.write_graph(sess.graph_def, model_path, 'raw_graph_def.pb', as_text=False)
    print("Saved Model")


def export_graph(model_path, env_name="env", target_nodes="action,predictions"):
    """
    Exports latest saved model to .bytes format for Unity embedding.
    :param model_path: path of model checkpoints.
    :param env_name: Name of associated Learning Environment.
    :param target_nodes: Comma separated string of needed output nodes for embedded graph.
    """
    ckpt = tf.train.get_checkpoint_state(model_path)
    freeze_graph.freeze_graph(input_graph=model_path + '/raw_graph_def.pb',
                              input_binary=True,
                              input_checkpoint=ckpt.model_checkpoint_path,
                              output_node_names=target_nodes,
                              output_graph=model_path + '/' + env_name + '.bytes',
                              clear_devices=True, initializer_nodes="", input_saver="",
                              restore_op_name="save/restore_all", filename_tensor_name="save/Const:0")

class QLearningModel(object):
    def __init__(self):
        self.normalize = False
        self.observation_in = []

    def create_global_steps(self):
        """Creates TF ops to track and increment global training step."""
        self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)
        self.increment_step = tf.assign(self.global_step, self.global_step + 1)

    def create_reward_encoder(self):
        """Creates TF ops to track and increment recent average cumulative reward."""
        self.last_reward = tf.Variable(0, name="last_reward", trainable=False, dtype=tf.float32)
        self.new_reward = tf.placeholder(shape=[], dtype=tf.float32, name='new_reward')
        self.update_reward = tf.assign(self.last_reward, self.new_reward)

    def create_visual_encoder(self, o_size_h, o_size_w, bw, h_size, num_streams, activation, num_layers):
        """
        Builds a set of visual (CNN) encoders.
        :param o_size_h: Height observation size.
        :param o_size_w: Width observation size.
        :param bw: Whether image is greyscale {True} or color {False}.
        :param h_size: Hidden layer size.
        :param num_streams: Number of visual streams to construct.
        :param activation: What type of activation function to use for layers.
        :return: List of hidden layer tensors.
        """
        if bw:
            c_channels = 1
        else:
            c_channels = 3

        self.observation_in.append(tf.placeholder(shape=[None, o_size_h, o_size_w, c_channels], dtype=tf.float32,
                                             name='observation_%d' % len(self.observation_in)))
        streams = []
        for i in range(num_streams):
            self.conv1 = tf.layers.conv2d(self.observation_in[-1], 16, kernel_size=[8, 8], strides=[4, 4],
                                          use_bias=False, activation=activation)
            self.conv2 = tf.layers.conv2d(self.conv1, 32, kernel_size=[4, 4], strides=[2, 2],
                                          use_bias=False, activation=activation)
            hidden = c_layers.flatten(self.conv2)
            for j in range(num_layers):
                hidden = tf.layers.dense(hidden, h_size, use_bias=False, activation=activation)
            streams.append(hidden)
        return streams

    def create_continuous_state_encoder(self, s_size, h_size, num_streams, activation, num_layers):
        """
        Builds a set of hidden state encoders.
        :param s_size: state input size.
        :param h_size: Hidden layer size.
        :param num_streams: Number of state streams to construct.
        :param activation: What type of activation function to use for layers.
        :return: List of hidden layer tensors.
        """
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32, name='state')

        if self.normalize:
            self.running_mean = tf.get_variable("running_mean", [s_size], trainable=False, dtype=tf.float32,
                                                initializer=tf.zeros_initializer())
            self.running_variance = tf.get_variable("running_variance", [s_size], trainable=False, dtype=tf.float32,
                                                    initializer=tf.ones_initializer())

            self.normalized_state = tf.clip_by_value((self.state_in - self.running_mean) / tf.sqrt(
                self.running_variance / (tf.cast(self.global_step, tf.float32) + 1)), -5, 5, name="normalized_state")

            self.new_mean = tf.placeholder(shape=[s_size], dtype=tf.float32, name='new_mean')
            self.new_variance = tf.placeholder(shape=[s_size], dtype=tf.float32, name='new_variance')
            self.update_mean = tf.assign(self.running_mean, self.new_mean)
            self.update_variance = tf.assign(self.running_variance, self.new_variance)
        else:
            self.normalized_state = self.state_in
        streams = []
        for i in range(num_streams):
            hidden = self.normalized_state
            for j in range(num_layers):
                hidden = tf.layers.dense(hidden, h_size, use_bias=False, activation=activation)
            streams.append(hidden)
        return streams

    def create_discrete_state_encoder(self, s_size, h_size, num_streams, activation, num_layers):
        """
        Builds a set of hidden state encoders from discrete state input.
        :param s_size: state input size (discrete).
        :param h_size: Hidden layer size.
        :param num_streams: Number of state streams to construct.
        :param activation: What type of activation function to use for layers.
        :return: List of hidden layer tensors.
        """
        self.state_in = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='state')
        state_in = tf.reshape(self.state_in, [-1])
        self.state_onehot = c_layers.one_hot_encoding(state_in, s_size)
        streams = []
        hidden = self.state_onehot
        if num_layers == 0:
            streams.append(hidden)
        for i in range(num_streams):
            for j in range(num_layers):
                hidden = tf.layers.dense(hidden, h_size, use_bias=False, activation=activation)
            streams.append(hidden)
        return streams

    def create_q_optimizer(self, lr, max_step):
        """
        Creates training-specific Tensorflow ops for DQN model update.
        :param lr: Learning rate
        :param max_step: Total number of training steps
        """
        self.loss = tf.reduce_mean(tf.squared_difference(self.action_value, self.targets))
        self.learning_rate = tf.train.polynomial_decay(lr, self.global_step,
                                                       max_step, 1e-10,
                                                       power=1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update_batch = optimizer.minimize(self.loss)

class DiscreteControlModel(QLearningModel):
    def __init__(self, lr, brain, h_size, max_step, normalize, num_layers):
        """
        Creates Discrete Control Q learning model.
        :param brain: State-space size
        :param h_size: Hidden layer size
        :param max_step: Total number of training steps
        :param normalize: Normalize observations
        :num_layers: Number of hidden layers in the value function approximator
        """
        super(DiscreteControlModel, self).__init__()
        self.create_global_steps()
        self.create_reward_encoder()
        self.normalize = normalize
        hidden_state, hidden_visual, hidden = None, None, None
        if brain.number_observations > 0:
            encoders = []
            for i in range(brain.number_observations):
                height_size, width_size = brain.camera_resolutions[i]['height'], brain.camera_resolutions[i]['width']
                bw = brain.camera_resolutions[i]['blackAndWhite']
                encoders.append(self.create_visual_encoder(height_size, width_size, bw, h_size, 1, tf.nn.elu, num_layers)[0])
            hidden_visual = tf.concat(encoders, axis=1)
        if brain.state_space_size > 0:
            s_size = brain.state_space_size
            if brain.state_space_type == "continuous":
                hidden_state = self.create_continuous_state_encoder(s_size, h_size, 1, tf.nn.elu, num_layers)[0]
            else:
                hidden_state = self.create_discrete_state_encoder(s_size, h_size, 1, tf.nn.elu, num_layers)[0]

        if hidden_visual is None and hidden_state is None:
            raise Exception("No valid network configuration possible. "
                            "There are no states or observations in this brain")
        elif hidden_visual is not None and hidden_state is None:
            hidden = hidden_visual
        elif hidden_visual is None and hidden_state is not None:
            self.hidden = hidden_state
        elif hidden_visual is not None and hidden_state is not None:
            hidden = tf.concat([hidden_visual, hidden_state], axis=1)
        a_size = brain.action_space_size

        # Value Function predictions
        self.predictions = tf.layers.dense(self.hidden, a_size, activation=None, use_bias=True,
                                     kernel_initializer=c_layers.variance_scaling_initializer(factor=1.0))
        self.predictions = tf.identity(self.predictions, name="predictions")
        self.output = tf.argmax(self.predictions,1)
        self.output = tf.identity(self.output, name="action")

        # Value Function update
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="targets")
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
        self.action_value = tf.reduce_sum(tf.multiply(self.predictions, self.actions_onehot), axis=1)

        self.create_q_optimizer(lr, max_step)
