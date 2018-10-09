from random import choice
import tensorflow as tf
import pandas as pd
from numpy import argmax, zeros

from .shallow_network import ShallowNetwork


class NeuralAgent(object):

    policies = {
        'greedy': {
            'action': lambda x: int(argmax(x)),
            'reward': max,
        }
    }

    def __init__(self, epsilon, action_space, **network_kwargs):
        self.action_space = action_space
        self.shallow_network = ShallowNetwork(self.action_space, **network_kwargs)
        self.shallow_network.initialize()
        self.policy = 'greedy'
        self.action_func = self.policies[self.policy]['action']
        self.reward_func = self.policies[self.policy]['reward']

    def train(self, memories):
        with tf.Session(graph=self.shallow_network.graph) as sess:
            reward_real = pd.DataFrame([m['reward'] for m in memories])
            reward_real_feed = {self.shallow_network.reward_real[installation]['generator']:
                                (reward_real[installation].values.reshape(-1, 1) if installation in reward_real.columns else zeros([len(memories), 1]))
                                for installation in self.shallow_network.reward_real}
            nodes = [
                self.shallow_network.update,
                self.shallow_network.loss,
            ]
            feed = {
                **{self.shallow_network.state: pd.DataFrame([m['state'] for m in memories])},
                **reward_real_feed,
            }
            sess.run([tf.global_variables_initializer()], feed_dict=feed)
            _, loss = sess.run(nodes, feed_dict=feed)

        return float(loss)

    def get_action(self, state, random):
        with tf.Session(graph=self.shallow_network.graph) as sess:
            reward_nodes = self.dict_to_list_of_tuples(self.shallow_network.reward_expected)
            nodes = [node[2] for node in reward_nodes]
            feed = {
                self.shallow_network.state: state,
            }
            sess.run([tf.global_variables_initializer()], feed_dict=feed)
            output = sess.run(nodes, feed_dict=feed)
            exp_rewards = {node[0]: {node[1]: reward.reshape(-1)} for node, reward in zip(reward_nodes, output)}
            if random:
                actions = self.get_random_action()
            else:
                actions = {node[0]: {node[1]: self.action_func(reward)}
                           for node, reward in zip(reward_nodes, output)}

            self.last_reward_expected = exp_rewards
            self.last_state = state
            self.last_actions = actions
            return actions

    def get_state_value(self, state):
        with tf.Session(graph=self.shallow_network.graph) as sess:
            reward_nodes = self.dict_to_list_of_tuples(self.shallow_network.reward_expected)
            nodes = [node[2] for node in reward_nodes]
            feed = {
                self.shallow_network.state: state,
            }
            sess.run([tf.global_variables_initializer()], feed_dict=feed)
            output = sess.run(nodes, feed)
            exp_rewards = {node[0]: self.reward_func(reward.reshape(-1))
                           for node, reward in zip(reward_nodes, output)}
            return exp_rewards

    def get_random_action(self):
        actions = {}
        for name, sub_space in self.action_space.items():
            actions[name] = {}
            for sub_space_name, sub_space_value in sub_space.items():
                actions[name][sub_space_name] = choice(sub_space_value)

        return actions

    @staticmethod
    def split_list(_list, n_splits=1):
        return [_list[i * len(_list) // n_splits: (i + 1) * len(_list) // n_splits] for i in range(n_splits)]
    
    @staticmethod
    def dict_to_list_of_tuples(_dict):
        lists = [[tuple([key]) + node for node in list(nodes.items())] for key, nodes in _dict.items()]
        return [elem for sublist in lists for elem in sublist]
