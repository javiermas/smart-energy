from random import choice
import tensorflow as tf
from numpy import argmax

from .shallow_network import ShallowNetwork


class NeuralAgent(object):

    policies = {
        'greedy': {
            'action': argmax,
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
            action_nodes = self.dict_to_list_of_tuples(self.shallow_network.actions)
            reward_nodes = self.dict_to_list_of_tuples(self.shallow_network.reward)
            #nodes = [node[2] for node in action_nodes + reward_nodes]
            nodes = [
                self.shallow_network.update
            ]
            feed = {
                self.shallow_network.state: state,
            }
            sess.run([tf.global_variables_initializer()], feed)
            output = sess.run(nodes, feed)
            actions, exp_rewards = self.split_list(output, 2)
            actions = {node[0]: {node[1]: action} for node, action in zip(action_nodes, actions)}
            exp_rewards = {node[0]: {node[1]: reward} for node, reward in zip(reward_nodes, exp_rewards)}
            return actions

    def get_action(self, state, random):
        with tf.Session(graph=self.shallow_network.graph) as sess:
            #action_nodes = self.dict_to_list_of_tuples(self.shallow_network.actions)
            reward_nodes = self.dict_to_list_of_tuples(self.shallow_network.expected_reward)
            nodes = [node[2] for node in reward_nodes]
            feed = {
                self.shallow_network.state: state,
            }
            sess.run([tf.global_variables_initializer()], feed)
            output = sess.run(nodes, feed)
            #actions, exp_rewards = self.split_list(output, 2)
            #actions = {node[0]: {node[1]: float(action)} for node, action in zip(action_nodes, actions)}
            exp_rewards = {node[0]: {node[1]: reward.reshape(-1)} for node, reward in zip(reward_nodes, output)}
            if random:
                actions = self.get_random_action()
            else:
                actions = {node[0]: {node[1]: self.action_func(reward)}
                           for node, reward in zip(reward_nodes, output)}

            self.last_expected_reward = exp_rewards
            self.last_state = state
            self.last_actions = actions
            return actions

    def get_state_value(self, state):
        with tf.Session(graph=self.shallow_network.graph) as sess:
            #action_nodes = self.dict_to_list_of_tuples(self.shallow_network.actions)
            reward_nodes = self.dict_to_list_of_tuples(self.shallow_network.expected_reward)
            nodes = [node[2] for node in reward_nodes]
            feed = {
                self.shallow_network.state: state,
            }
            sess.run([tf.global_variables_initializer()], feed)
            output = sess.run(nodes, feed)
            #actions, exp_rewards = self.split_list(output, 2)
            #actions = {node[0]: {node[1]: float(action)} for node, action in zip(action_nodes, actions)}
            exp_rewards = {node[0]: {node[1]: reward.reshape(-1)} for node, reward in zip(reward_nodes, output)}
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
