from random import choice
import tensorflow as tf
from .shallow_network import ShallowNetwork


class BasicAgent(object):

    def __init__(self, epsilon, action_space, **network_kwargs):
        self.action_space = action_space
        self.shallow_network = ShallowNetwork(self.action_space, **network_kwargs)
        self.shallow_network.initialize()

    def get_action(self, state):
        with tf.Session(graph=self.shallow_network.graph) as sess:
            action_nodes = [node for nodes in self.shallow_network.actions.values()
                            for node in nodes.values()]
            reward_nodes = [node for nodes in self.shallow_network.reward.values()
                            for node in nodes.values()]
            nodes = action_nodes + reward_nodes
            feed = {
                self.shallow_network.state: state,
            }
            sess.run([tf.global_variables_initializer()], feed)
            output = sess.run(nodes, feed)
            actions, expected_rewards = self.split_list(output, 2)
            return actions

    def get_random_action(self):
        actions = {}
        for name, sub_space in self.action_space.items():
            actions[name] = {}
            for sub_space_name, sub_space_value in sub_space.items():
                actions[name][sub_space_name] = choice(range(sub_space_value))

        return actions
    
    @staticmethod
    def split_list(_list, n_splits=1):
        return [_list[i * len(_list) // n_splits: (i + 1) * len(_list) // n_splits] for i in range(n_splits)]
