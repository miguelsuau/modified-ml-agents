# # Unity ML Agents
# ## ML-Agent Learning (DQN)
# Contains an implementation of DQN

import logging
import os

import numpy as np
import tensorflow as tf

from unityagents import AllBrainInfo
from unitytrainers.buffer import Buffer
from unitytrainers.dqn.DQNmodel import DQNModel
from unitytrainers.madqn.model import MADQNModel
from unitytrainers.trainer import UnityTrainerException, Trainer

logger = logging.getLogger("unityagents")

history_keys = ['states', 'actions', 'rewards', 'next_states', 'done']


class MADQNTrainer(Trainer):
    """The DQNTrainer is an implementation of the DQN algorithm."""

    def __init__(self, sess, env, brain_name, trainer_parameters, training, seed):
        """
        Responsible for collecting experiences and training DQN model.
        :param sess: Tensorflow session.
        :param env: The UnityEnvironment.
        :param  trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        """
        self.param_keys = ['batch_size', 'replay_memory_size', 'epsilon_start', 'epsilon_end', 'epsilon_decay_steps',
                           'gamma', 'hidden_units', 'lambd', 'learning_rate', 'max_steps', 'tau', 'update_freq',
                           'normalize', 'num_layers','summary_freq', 'use_recurrent','graph_scope', 'summary_path',
                           'pre_train_steps', 'frozen', 'update_frozen_freq']

        for k in self.param_keys:
            if k not in trainer_parameters:
                raise UnityTrainerException("The hyperparameter {0} could not be found for the DQN trainer of "
                                            "brain {1}.".format(k, brain_name))

        super(MADQNTrainer, self).__init__(sess, env, brain_name, trainer_parameters, training)

        self.use_recurrent = trainer_parameters["use_recurrent"]
        self.sequence_length = 1
        self.m_size = None
        if self.use_recurrent:
            self.m_size = trainer_parameters["memory_size"]
            self.sequence_length = trainer_parameters["sequence_length"]
        if self.use_recurrent:
            if self.m_size == 0:
                raise UnityTrainerException("The memory size for brain {0} is 0 even though the trainer uses recurrent."
                                            .format(brain_name))
            elif self.m_size % 4 != 0:
                raise UnityTrainerException("The memory size for brain {0} is {1} but it must be divisible by 4."
                                            .format(brain_name, self.m_size))

        self.variable_scope = trainer_parameters['graph_scope']
        with tf.variable_scope(self.variable_scope):
            tf.set_random_seed(seed)
            self.main = MADQNModel(env.brains[brain_name],
                                  lr=float(trainer_parameters['learning_rate']),
                                  h_size=int(trainer_parameters['hidden_units']),
                                  epsilon_start=float(trainer_parameters['epsilon_start']),
                                  epsilon_end=float(trainer_parameters['epsilon_end']),
                                  epsilon_decay_steps=float(trainer_parameters['epsilon_decay_steps']),
                                  tau=float(trainer_parameters['tau']),
                                  max_step=float(trainer_parameters['max_steps']),
                                  normalize=trainer_parameters['normalize'],
                                  use_recurrent=trainer_parameters['use_recurrent'],
                                  num_layers=int(trainer_parameters['num_layers']),
                                  m_size=self.m_size,
                                  frozen=trainer_parameters['frozen'],
                                  update_frozen_freq=trainer_parameters['update_frozen_freq'])
            self.target = MADQNModel(env.brains[brain_name],
                                  lr=float(trainer_parameters['learning_rate']),
                                  h_size=int(trainer_parameters['hidden_units']),
                                  epsilon_start=float(trainer_parameters['epsilon_start']),
                                  epsilon_end=float(trainer_parameters['epsilon_end']),
                                  epsilon_decay_steps=float(trainer_parameters['epsilon_decay_steps']),
                                  tau=float(trainer_parameters['tau']),
                                  max_step=float(trainer_parameters['max_steps']),
                                  normalize=trainer_parameters['normalize'],
                                  use_recurrent=trainer_parameters['use_recurrent'],
                                  num_layers=int(trainer_parameters['num_layers']),
                                  m_size=self.m_size,
                                  frozen=trainer_parameters['frozen'],
                                  update_frozen_freq=trainer_parameters['update_frozen_freq'])


        stats = {'cumulative_reward': [], 'episode_length': [], 'value_estimate': [],
                 'learning_rate': [], 'epsilon': []}
        self.stats = stats

        self.training_buffer = Buffer()
        self.cumulative_rewards = {}
        self.episode_steps = {}
        self.is_continuous = (env.brains[brain_name].vector_action_space_type == "continuous")
        self.use_observations = (env.brains[brain_name].number_visual_observations > 0)
        self.use_states = (env.brains[brain_name].vector_observation_space_size > 0)
        self.summary_path = trainer_parameters['summary_path']
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

        self.summary_writer = tf.summary.FileWriter(self.summary_path)

    def __str__(self):
        return '''Hypermarameters for the DQN Trainer of brain {0}: \n{1}'''.format(
            self.brain_name, '\n'.join(['\t{0}:\t{1}'.format(x, self.trainer_parameters[x]) for x in self.param_keys]))

    @property
    def parameters(self):
        """
        Returns the trainer parameters of the trainer.
        """
        return self.trainer_parameters

    @property
    def graph_scope(self):
        """
        Returns the graph scope of the trainer.
        """
        return self.variable_scope

    @property
    def get_max_steps(self):
        """
        Returns the maximum number of steps. Is used to know when the trainer should be stopped.
        :return: The maximum number of steps of the trainer
        """
        return float(self.trainer_parameters['max_steps'])

    @property
    def get_step(self):
        """
        Returns the number of steps the trainer has performed
        :return: the step count of the trainer
        """
        return self.sess.run(self.main.global_step)

    @property
    def get_last_reward(self):
        """
        Returns the last reward the trainer has had
        :return: the new last reward
        """
        return self.sess.run(self.main.last_reward)

    def increment_step(self):
        """
        Increment the step count of the trainer
        """
        self.sess.run(self.main.increment_step)

    def update_last_reward(self):
        """
        Updates the last reward
        """
        if len(self.stats['cumulative_reward']) > 0:
            mean_reward = np.mean(self.stats['cumulative_reward'])
            self.sess.run(self.main.update_reward, feed_dict={self.main.new_reward: mean_reward})

    def running_average(self, data, steps, running_mean, running_variance):
        """
        Computes new running mean and variances.
        :param data: New piece of data.
        :param steps: Total number of data so far.
        :param running_mean: TF op corresponding to stored running mean.
        :param running_variance: TF op corresponding to stored running variance.
        :return: New mean and variance values.
        """
        mean, var = self.sess.run([running_mean, running_variance])
        current_x = np.mean(data, axis=0)
        new_mean = mean + (current_x - mean) / (steps + 1)
        new_variance = var + (current_x - new_mean) * (current_x - mean)
        return new_mean, new_variance

    def take_action(self, all_brain_info: AllBrainInfo):
        """
        Decides actions given state/observation information, and takes them in environment.
        :param all_brain_info: A dictionary of brain names and BrainInfo from environment.
        :return: a tuple containing action, memories, values and an object
        to be passed to add experiences
        """
        self.steps = self.get_step
        curr_brain_info = all_brain_info[self.brain_name]
        if len(curr_brain_info.agents) == 0:
            return [], [], [], None
        feed_dict = {}
        run_list = [self.main.predictions, self.main.chosen_action, self.main.epsilon]
        if not self.trainer_parameters['frozen']:
            run_list += [self.main.learning_rate]
        if self.use_observations:
            for i, _ in enumerate(curr_brain_info.visual_observations):
                feed_dict[self.main.visual_in[i]] = curr_brain_info.visual_observations[i]
        if self.use_states:
            feed_dict[self.main.vector_in] = curr_brain_info.vector_observations
        if self.use_recurrent:
            if curr_brain_info.memories.shape[1] == 0:
                curr_brain_info.memories = np.zeros((len(curr_brain_info.agents), self.m_size))
            feed_dict[self.main.memory_in] = curr_brain_info.memories
            run_list += [self.main.memory_out]
        if (self.is_training and self.brain.vector_observation_space_type == "continuous" and
                self.use_states and self.trainer_parameters['normalize']):
            new_mean, new_variance = self.running_average(
                curr_brain_info.vector_observations, self.steps, self.main.running_mean, self.main.running_variance)
            feed_dict[self.main.new_mean] = new_mean
            feed_dict[self.main.new_variance] = new_variance
            run_list = run_list + [self.main.update_mean, self.main.update_variance]

        values = self.sess.run(run_list, feed_dict=feed_dict)
        run_out = dict(zip(run_list, values))
        self.stats['value_estimate'].append(run_out[self.main.predictions].mean())
        self.stats['epsilon'].append(run_out[self.main.epsilon])
        if not self.trainer_parameters['frozen']:
            self.stats['learning_rate'].append(run_out[self.main.learning_rate])
        if self.use_recurrent:
            return (run_out[self.main.chosen_action],
                    run_out[self.main.memory_out],
                    [str(v) for v in run_out[self.main.value]],
                    run_out)
        else:
            return (run_out[self.main.chosen_action],
                    None,
                    [str(v) for v in run_out[self.main.predictions]],
                    run_out)

    def add_experiences(self, curr_all_info: AllBrainInfo, next_all_info: AllBrainInfo, take_action_outputs):

        """
        Adds experiences to each agent's experience history.
        :param curr_all_info: Dictionary of all current brains and corresponding BrainInfo.
        :param next_all_info: Dictionary of all current brains and corresponding BrainInfo.
        :param take_action_outputs: The outputs of the take action method.
        """
        curr_info = curr_all_info[self.brain_name]
        next_info = next_all_info[self.brain_name]

        for agent_id in curr_info.agents:
            self.training_buffer[agent_id].last_brain_info = curr_info
            self.training_buffer[agent_id].last_take_action_outputs = take_action_outputs

        try:
            self.history_dict
        except:
            if len(curr_info.agents) > 0:
                self.create_history(curr_info)

        if not self.trainer_parameters['frozen']:
            if len(curr_info.agents) > 0:
                for agent_id in next_info.agents:
                    stored_info = self.training_buffer[agent_id].last_brain_info
                    stored_take_action_outputs = self.training_buffer[agent_id].last_take_action_outputs
                    if stored_info is None:
                        continue
                    else:
                        idx = stored_info.agents.index(agent_id)
                        next_idx = next_info.agents.index(agent_id)
                        if not stored_info.local_done[idx]:
                            if self.use_observations:
                                for i, _ in enumerate(info.observations):
                                    self.history_dict[agent_id]['observations%d' % i].append([stored_info.visual_observations[i][idx]])
                                    self.history_dict[agent_id]['next_observations%d' % i].append([next_info.visual_observations[i][next_idx]])
                            if self.use_states:
                                self.history_dict[agent_id]['states'].append(stored_info.vector_observations[idx])
                                self.history_dict[agent_id]['next_states'].append(next_info.vector_observations[next_idx])
                            actions = stored_take_action_outputs[self.main.chosen_action]
                            self.history_dict[agent_id]['actions'].append(actions[idx])
                            self.history_dict[agent_id]['rewards'].append(next_info.rewards[next_idx])
                            self.history_dict[agent_id]['done'].append(next_info.local_done[next_idx])
                            if agent_id not in self.cumulative_rewards:
                                self.cumulative_rewards[agent_id] = 0
                            self.cumulative_rewards[agent_id] += next_info.rewards[next_idx]
                            if agent_id not in self.episode_steps:
                                self.episode_steps[agent_id] = 0
                            self.episode_steps[agent_id] += 1
                            #print("local buffer " + str(len(self.training_buffer[agent_id]['actions'])))

        else:
            for agent_id in next_info.agents:
                stored_info = self.training_buffer[agent_id].last_brain_info
                stored_take_action_outputs = self.training_buffer[agent_id].last_take_action_outputs
                if stored_info is None:
                    continue
                else:
                    idx = stored_info.agents.index(agent_id)
                    next_idx = next_info.agents.index(agent_id)
                    if not stored_info.local_done[idx]:
                        if agent_id not in self.cumulative_rewards:
                            self.cumulative_rewards[agent_id] = 0
                        self.cumulative_rewards[agent_id] += next_info.rewards[next_idx]
                        if agent_id not in self.episode_steps:
                            self.episode_steps[agent_id] = 0
                        self.episode_steps[agent_id] += 1


    def process_experiences(self, all_info: AllBrainInfo):
        """
        Add description
        """
        if not self.trainer_parameters['frozen']:
            info = all_info[self.brain_name]
            for l in range(len(info.agents)):
                agent_id = info.agents[l]
                if info.local_done[l]  and len(self.history_dict[agent_id]['actions']) > 0:
                    history = self.history_dict[agent_id]
                    self.training_buffer.append_replay_memory(local_buffer=history, replay_memory_size=self.trainer_parameters['replay_memory_size'])
                    self.empty_local_history(agent_id)

                    if info.local_done[l]:
                        self.stats['cumulative_reward'].append(self.cumulative_rewards[agent_id])
                        self.stats['episode_length'].append(self.episode_steps[agent_id])
                        self.cumulative_rewards[agent_id] = 0
                        self.episode_steps[agent_id] = 0
        else:
            info = all_info[self.brain_name]
            for l in range(len(info.agents)):
                agent_id = info.agents[l]
                if info.local_done[l]:
                    self.stats['cumulative_reward'].append(self.cumulative_rewards[agent_id])
                    self.stats['episode_length'].append(self.episode_steps[agent_id])
                    self.cumulative_rewards[agent_id] = 0
                    self.episode_steps[agent_id] = 0

    def empty_local_history(self, agent_id):
        """
        Empties the experience history for a single agent.
        :param agent_dict: Dictionary of agent experience history.
        :return: Emptied dictionary (except for cumulative_reward and episode_steps).
        """
        for key in history_keys:
            self.history_dict[agent_id][key] = []
        for i, _ in enumerate(key for key in self.history_dict[agent_id].keys() if key.startswith('observations')):
            self.history_dict[agent_id]['observations%d' % i] = []
        for i, _ in enumerate(key for key in self.history_dict[agent_id].keys() if key.startswith('next_observations')):
            self.history_dict[agent_id]['next_observations%d' % i] = []

    def create_history(self, agent_info):
        """
        Clears all agent histories and resets reward and episode length counters.
        :param agent_info: a BrainInfo object.
        :return: an emptied history dictionary.
        """
        self.history_dict = {}
        for agent_id in agent_info.agents:
            self.history_dict[agent_id] = {}
            self.empty_local_history(agent_id)
            #print(self.history_dict[agent_id])
            for i, _ in enumerate(agent_info.visual_observations):
                self.history_dict[agent_id]['observations%d' % i] = []

    def end_episode(self):
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets.
        """
        self.training_buffer.reset_all()
        for agent_id in self.cumulative_rewards:
            self.cumulative_rewards[agent_id] = 0
        for agent_id in self.episode_steps:
            self.episode_steps[agent_id] = 0

    def is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not update_model() can be run
        """
        return (len(self.training_buffer.update_buffer['actions']) > \
                int(self.trainer_parameters['pre_train_steps']) or \
                self.trainer_parameters['frozen']) and \
                self.steps % int(self.trainer_parameters['update_freq']) == 0

    def sample(self, batch_size):
        """
        Samples training batch from experience buffer
        :param batch_size: Size of the training batch
        """
        self.update_batch = {}
        _buffer = self.training_buffer.update_buffer
        idx = np.random.choice(len(_buffer['actions']), batch_size)
        for key in _buffer.keys():
            self.update_batch[key] = []
            if len(_buffer[key]) > batch_size:
                for x in idx:
                    self.update_batch[key].append( _buffer[key][x])

    def update_model(self):
        """
        Uses training_buffer to update model.
        """
        if not self.trainer_parameters['frozen']:
            self.sample(self.trainer_parameters['batch_size'])
            feed_dict1 = {}
            feed_dict2 = {}
            feed_dict3 = {}

            if self.use_states:
                if self.brain.vector_observation_space_type == "continuous":
                    feed_dict1[self.main.vector_in] = self.update_batch['next_states']#.reshape(
                        #[-1, self.brain.vector_observation_space_size * self.brain.num_stacked_vector_observations])
                    feed_dict2[self.target.vector_in] = self.update_batch['next_states']#.reshape(
                        #[-1, self.brain.vector_observation_space_size * self.brain.num_stacked_vector_observations])
                    feed_dict3[self.main.vector_in] = self.update_batch['states']#.reshape(
                        #[-1, self.brain.vector_observation_space_size * self.brain.num_stacked_vector_observations])
                else:
                    feed_dict1[self.main.vector_in] = self.update_batch['next_states'].reshape([-1, self.brain.num_stacked_vector_observations])
                    feed_dict2[self.target.vector_in] = self.update_batch['next_states'].reshape([-1, self.brain.num_stacked_vector_observations])
                    feed_dict3[self.main.vector_in] = self.update_batch['states'].reshape([-1, self.brain.num_stacked_vector_observations])
            if self.use_observations:
                for i, _ in enumerate(self.main.visual_in):
                    _obs = self.update_batch['next_observations%d' % i]
                    (_batch, _seq, _w, _h, _c) = _obs.shape
                    feed_dict1[self.main.visual_in[i]] = _obs.reshape([-1, _w, _h, _c])
                    feed_dict2[self.target.visual_in[i]] = _obs.reshape([-1, _w, _h, _c])
                    _obs = self.update_batch['observations%d' % i]
                    (_batch, _seq, _w, _h, _c) = _obs.shape
                    feed_dict3[self.main.visual_in[i]] = _obs.reshape([-1, _w, _h, _c])

            # Find best action for each state according to Q1 (main model)
            Q1_actions = self.sess.run(self.main.output, feed_dict1)
            # Double Q-learning:
            feed_dict2[self.target.actions] = Q1_actions[:,0]
            # Find Q2 (target model) value of best action according to Q1
            Q2_values = self.sess.run(self.target.action_value, feed_dict2)
            inverse_done = np.invert(self.update_batch['done'])
            targets = self.update_batch['rewards'] + self.trainer_parameters['gamma']*Q2_values*inverse_done
            feed_dict3[self.main.targets] = targets
            feed_dict3[self.main.actions] = np.array(self.update_batch['actions'])[:,0]
            # Update main model using the calculated targets
            self.sess.run(self.main.update_batch, feed_dict= feed_dict3)
            # Update target model toward main model
            for op in self.op_holder:
                self.sess.run(op)
            # Empty replay memory
            if self.steps % self.trainer_parameters['update_frozen_freq'] == 0 and self.steps != 0:
                self.training_buffer.reset_replay_memory()
                self.sess.run(self.main.increment_updates)
        else:
            if self.steps % self.trainer_parameters['update_frozen_freq'] == 0 and self.steps != 0:
                for update in self.update_frozen_brain:
                    self.sess.run(update)

    def write_summary(self, lesson_number):
        """
        Saves training statistics to Tensorboard.
        :param lesson_number: The lesson the trainer is at.
        """
        if (self.steps % self.trainer_parameters['summary_freq'] == 0 and self.steps != 0 and
                self.is_training and self.steps <= self.get_max_steps):
            print(len(self.training_buffer.update_buffer['actions']))
            if len(self.stats['cumulative_reward']) > 0:
                mean_reward = np.mean(self.stats['cumulative_reward'])
                logger.info(" {}: Step: {}. Mean Reward: {:0.3f}. Std of Reward: {:0.3f}."
                            .format(self.brain_name, self.steps, mean_reward, np.std(self.stats['cumulative_reward'])))
            summary = tf.Summary()
            for key in self.stats:
                if len(self.stats[key]) > 0:
                    stat_mean = float(np.mean(self.stats[key]))
                    summary.value.add(tag='Info/{}'.format(key), simple_value=stat_mean)
                    self.stats[key] = []
            summary.value.add(tag='Info/Lesson', simple_value=lesson_number)
            self.summary_writer.add_summary(summary, self.steps)
            self.summary_writer.flush()


    def update_target_graph(self, tfVars):
        total_vars = len(tfVars)
        self.op_holder = []
        for idx,var in enumerate(tfVars[0:total_vars//2]):
            self.op_holder.append(
                tfVars[idx+total_vars//2].assign((
                var.value()*self.trainer_parameters['tau'])
                + ((1-self.trainer_parameters['tau'])*tfVars[idx+total_vars//2].value())))

    def update_frozen_brain_graph(self, frozen_brain_vars, free_brain_vars):
        self.update_frozen_brain = []
        for idx,var in enumerate(free_brain_vars):
            self.update_frozen_brain.append(
                frozen_brain_vars[idx].assign(var.value()))

def discount_rewards(r, gamma=0.99, value_next=0.0):
    """
    Computes discounted sum of future rewards for use in updating value estimate.
    :param r: List of rewards.
    :param gamma: Discount factor.
    :param value_next: T+1 value estimate for returns calculation.
    :return: discounted sum of future rewards as list.
    """
    discounted_r = np.zeros_like(r)
    running_add = value_next
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r
