import numpy as np
import random

history_keys = ['states', 'actions', 'rewards', 'next_states', 'done']


def empty_local_history(agent_dict):
    """
    Empties the experience history for a single agent.
    :param agent_dict: Dictionary of agent experience history.
    :return: Emptied dictionary (except for cumulative_reward and episode_steps).
    """
    for key in history_keys:
        agent_dict[key] = []
    for i, _ in enumerate(key for key in agent_dict.keys() if key.startswith('observations')):
        agent_dict['observations%d' % i] = []
    for i, _ in enumerate(key for key in agent_dict.keys() if key.startswith('next_observations')):
        agent_dict['next_observations%d' % i] = []
    return agent_dict


def vectorize_history(agent_dict):
    """
    Converts dictionary of lists into dictionary of numpy arrays.
    :param agent_dict: Dictionary of agent experience history.
    :return: dictionary of numpy arrays.
    """
    for key in history_keys:
        agent_dict[key] = np.array(agent_dict[key])
    for key in (key for key in agent_dict.keys() if key.startswith('observations')):
        agent_dict[key] = np.array(agent_dict[key])
    return agent_dict


def empty_all_history(agent_info):
    """
    Clears all agent histories and resets reward and episode length counters.
    :param agent_info: a BrainInfo object.
    :return: an emptied history dictionary.
    """
    history_dict = {}
    for agent in agent_info.agents:
        history_dict[agent] = {}
        history_dict[agent] = empty_local_history(history_dict[agent])
        history_dict[agent]['cumulative_reward'] = 0
        history_dict[agent]['episode_steps'] = 0
        for i, _ in enumerate(agent_info.observations):
            history_dict[agent]['observations%d' % i] = []
    return history_dict

def append_replay_memory(global_buffer, local_buffer):
    """
    Appends the buffer of an agent to the update buffer.
    :param agent_id: The id of the agent which data will be appended
    :param key_list: The fields that must be added. If None: all fields will be appended.
    """
    for key in history_keys:
        print(np.shape(global_buffer[key]))
        print(key)
        global_buffer[key] = np.concatenate([global_buffer[key], local_buffer[key]], axis=0)
    return global_buffer

def set_history(global_buffer, local_buffer=None):
    """
    Creates new global_buffer from existing local_buffer
    :param global_buffer: Global buffer for all agents experiences.
    :param local_buffer: Local history for individual agents experiences.
    :return: Global buffer with new experiences.
    """
    for key in history_keys:
        global_buffer[key] = np.copy(local_buffer[key])
    for key in (key for key in local_buffer.keys() if key.startswith('observations')):
        global_buffer[key] = np.array(local_buffer[key])
    return global_buffer

def delete_entries_replay_memory(global_buffer, n):
    """
    Delete n random entries in history buffer to leave room for new experiences
    :param global_buffer: Global buffer for all agents experiences.
    :param n: Number of entries to delete
    :return: Global buffer with first experience removed.
    """
    idx =  random.sample(range(len(global_buffer['actions'])), n)
    #idx = range(n)
    for key in history_keys:
        global_buffer[key] = np.delete(global_buffer[key], idx, 0)
    return global_buffer
