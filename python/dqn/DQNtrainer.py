import numpy as np
import random
import tensorflow as tf

from dqn.DQNhistory import *


class Trainer(object):
    def __init__(self, DQmain, DQtarget, sess, info, is_continuous, use_observations, use_states, replay_memory_size, training):
        """
        Responsible for collecting experiences and training PPO model.
        :param ppo_model: Tensorflow graph defining model.
        :param sess: Tensorflow session.
        :param info: Environment BrainInfo object.
        :param is_continuous: Whether action-space is continuous.
        :param use_observations: Whether agent takes image observations.
        """
        self.main = DQmain
        self.target = DQtarget
        self.sess = sess
        stats = {'cumulative_reward': [], 'episode_length': [], 'q_value_estimate': [], 'learning_rate': [], 'epsilon': []}
        self.stats = stats
        self.is_training = training
        self.reset_buffers(info, total=True)
        self.training_buffer = vectorize_history(empty_local_history({}))
        self.is_continuous = is_continuous
        self.use_observations = use_observations
        self.use_states = use_states
        self.replay_memory_size = replay_memory_size

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

    def policy(self, a_size, output, epsilon):
        """
        Creates epsilon-greedy policy
        :param a_size: Action space Size
        :param output: Action with the highest value according to the model
        :param epsilon: Probability of random action
        """
        prob = np.ones(a_size)*epsilon/a_size
        prob[output] += 1 - epsilon
        return prob

    def take_action(self, info, env, brain_name, steps, normalize, epsilon, a_size):
        # epsilon should be handled as a tf variable
        """
        Decides actions given state/observation information, and takes them in environment.
        :param info: Current BrainInfo from environment.
        :param env: Environment to take actions in.
        :param brain_name: Name of brain we are learning model for.
        :return: BrainInfo corresponding to new environment state.
        """
        epsi = None
        #feed_dict = {self.main.batch_size: len(info.states)}
        feed_dict = {}
        run_list = [self.main.output, self.main.predictions]
        if self.is_continuous:
            epsi = np.random.randn(len(info.states), env.brains[brain_name].action_space_size)
            feed_dict[self.main.epsilon] = epsi
        if self.use_observations:
            for i, _ in enumerate(info.observations):
                feed_dict[self.main.observation_in[i]] = info.observations[i]
        if self.use_states:
            feed_dict[self.main.state_in] = info.states
        if self.is_training and env.brains[brain_name].state_space_type == "continuous" and self.use_states and normalize:
            new_mean, new_variance = self.running_average(info.states, steps, self.main.running_mean,
                                                          self.main.running_variance)
            feed_dict[self.main.new_mean] = new_mean
            feed_dict[new_variance] = new_variance
            run_list = run_list + [self.main.update_mean, self.main.update_variance]
            predictions, learn_rate, _, _ = self.sess.run(run_list, feed_dict=feed_dict)
        else:
        #    q_pred, learn_rate = self.sess.run(run_list, feed_dict=feed_dict)
        # create epsilon greedy policy
            output, predictions = self.sess.run(run_list, feed_dict=feed_dict)

        prob = self.policy(a_size, output, epsilon)
        # choose action
        action = np.random.choice(np.arange(a_size), p=prob) # inlcude this operation in the tf graph
        self.stats['q_value_estimate'].append(predictions[:, action])
        self.stats['epsilon'].append(epsilon)
        #self.stats['learning_rate'].append(learn_rate)
        new_info = env.step(action, value={brain_name: predictions[:,action]})[brain_name]
        self.add_experiences(info, new_info, action)
        return new_info

    def add_experiences(self, info, next_info, actions):
        """
        Adds experiences to each agent's experience history.
        :param info: Current BrainInfo.
        :param next_info: Next BrainInfo.
        :param epsi: Epsilon value (for continuous control)
        :param actions: Chosen actions.
        :param a_dist: Action probabilities.
        :param value: Value estimates.
        """
        for (agent, history) in self.history_dict.items():
            if agent in info.agents:
                idx = info.agents.index(agent)
                if not info.local_done[idx]:
                    if self.use_observations:
                        for i, _ in enumerate(info.observations):
                            history['observations%d' % i].append([info.observations[i][idx]])
                            history['next_observations%d' % i].append([next_info.observations[i][idx]])
                    if self.use_states:
                        history['states'].append(info.states[idx])
                        history['next_states'].append(next_info.states[idx])
                    #history['actions'].append(actions[idx])
                    history['actions'].append(actions) # change this so it looks like the line above
                    history['rewards'].append(next_info.rewards[idx])
                    history['done'].append(next_info.local_done[idx])
                    history['cumulative_reward'] += next_info.rewards[idx]
                    history['episode_steps'] += 1


    def store_experiences(self, info):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param info: Current BrainInfo
        :param time_horizon: Max steps for individual agent history before processing.
        :param gamma: Discount factor.
        :param lambd: GAE factor.
        """
        for l in range(len(info.agents)):
            if info.local_done[l]  and len(self.history_dict[info.agents[l]]['actions']) > 0:
                history = vectorize_history(self.history_dict[info.agents[l]])
                if len(self.training_buffer['actions']) > self.replay_memory_size:
                    delete_entries_history(global_buffer=self.training_buffer,n=len(history['actions']))
                if len(self.training_buffer['actions']) > 0:
                    append_history(global_buffer=self.training_buffer, local_buffer=history)
                else:
                    set_history(global_buffer=self.training_buffer, local_buffer=history)
                self.history_dict[info.agents[l]] = empty_local_history(self.history_dict[info.agents[l]])
                if info.local_done[l]:
                    self.stats['cumulative_reward'].append(history['cumulative_reward'])
                    self.stats['episode_length'].append(history['episode_steps'])
                    history['cumulative_reward'] = 0
                    history['episode_steps'] = 0

    def reset_buffers(self, brain_info=None, total=False):
        """
        Resets either all training buffers or local training buffers
        :param brain_info: The BrainInfo object containing agent ids.
        :param total: Whether to completely clear buffer.
        """
        if not total:
            for key in self.history_dict:
                self.history_dict[key] = empty_local_history(self.history_dict[key])
        else:
            self.history_dict = empty_all_history(agent_info=brain_info)

    def sample(self, batch_size):
        """
        Samples training batch from experience buffer
        :param batch_size: Size of the training batch
        """
        self.training_batch = {}
        idx = np.random.choice(len(self.training_buffer['actions']), batch_size)
        for key, values in self.training_buffer.items():
            if len(values) > batch_size:
                self.training_batch[key] = values[idx]

    def update_model(self, batch_size, gamma, epsilon, a_size):
        """
        Uses training_buffer to update model.
        :param batch_size: Size of each mini-batch update.
        :param gamma: Discount rate
        :param epsilon: Probabilty of random action
        :a_size: Action space size
        """
        self.sample(batch_size)
        feed_dict1 = {}
        feed_dict2 = {}
        feed_dict3 = {}
        if self.use_states:
            feed_dict1[self.main.state_in] = np.vstack(self.training_batch['next_states'])
            feed_dict2[self.target.state_in] = np.vstack(self.training_batch['next_states'])
            feed_dict3[self.main.state_in] = np.vstack(self.training_batch['states'])
        if self.use_observations:
            for i, _ in enumerate(self.main.observation_in):
                feed_dict1[self.main.observation_in[i]] = np.vstack(self.training_batch['next_observations%d' % i])
                feed_dict2[self.target.observation_in[i]] = np.vstack(self.training_batch['next_observations%d' % i])
                feed_dict3[self.main.observation_in[i]] = np.vstack(self.training_batch['observations%d' % i])

        # Find best action for each state according to Q1 (main model)
        Q1_actions = self.sess.run(self.main.output, feed_dict1)
        # Double Q-learning:
        feed_dict2[self.target.actions] = Q1_actions
        # Find Q2 (target model) value of best action according to Q1
        Q2_values = self.sess.run(self.target.action_value, feed_dict2)

        inverse_done = np.invert(self.training_batch['done'])
        targets = self.training_batch['rewards'] + gamma*Q2_values*inverse_done
        feed_dict3[self.main.targets] = targets
        feed_dict3[self.main.actions] = self.training_batch['actions']

        # Update main model using the calculated targets
        self.sess.run(self.main.update_batch, feed_dict= feed_dict3)

    def write_summary(self, summary_writer, steps, lesson_number):
        """
        Saves training statistics to Tensorboard.
        :param summary_writer: writer associated with Tensorflow session.
        :param steps: Number of environment steps in training process.
        """
        if len(self.stats['cumulative_reward']) > 0:
            mean_reward = np.mean(self.stats['cumulative_reward'])
            print("Step: {0}. Mean Reward: {1}. Std of Reward: {2}."
                  .format(steps, mean_reward, np.std(self.stats['cumulative_reward'])))
        summary = tf.Summary()
        for key in self.stats:
            if len(self.stats[key]) > 0:
                stat_mean = float(np.mean(self.stats[key]))
                summary.value.add(tag='Info/{}'.format(key), simple_value=stat_mean)
                self.stats[key] = []
        summary.value.add(tag='Info/Lesson', simple_value=lesson_number)
        summary_writer.add_summary(summary, steps)
        summary_writer.flush()

    def write_text(self, summary_writer, key, input_dict, steps):
        """
        Saves text to Tensorboard.
        Note: Only works on tensorflow r1.2 or above.
        :param summary_writer: writer associated with Tensorflow session.
        :param key: The name of the text.
        :param input_dict: A dictionary that will be displayed in a table on Tensorboard.
        :param steps: Number of environment steps in training process.
        """
        try:
            s_op = tf.summary.text(key,
                    tf.convert_to_tensor(([[str(x), str(input_dict[x])] for x in input_dict]))
                    )
            s = self.sess.run(s_op)
            summary_writer.add_summary(s, steps)
        except:
            print("Cannot write text summary for Tensorboard. Tensorflow version must be r1.2 or above.")
            pass

    def update_target_graph(self, tfVars, tau):
        total_vars = len(tfVars)
        op_holder = []
        for idx,var in enumerate(tfVars[0:total_vars//2]):
            op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
        return op_holder

    def update_target(self, op_holder):
        for op in op_holder:
            self.sess.run(op)
