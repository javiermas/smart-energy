from random import choice
import tensorflow as tf
from .shallow_network import ShallowNetwork

class BasicAgent(object):

    def __init__(self, epsilon, action_space, state_size, **network_kwargs):
        self.action_space = action_space
        self.state_size = state_size
        self.shallow_network = ShallowNetwork(self.action_space, state_size, **network_kwargs)
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
            actions, expected_rewards = actions, expected_rewards = sess.run(nodes, feed)

    def get_random_action(self):
        actions = {}
        for name, sub_space in self.action_space.items():
            actions[name] = {}
            for sub_space_name, sub_space_value in sub_space.items():
                actions[name][sub_space_name] = choice(range(sub_space_value))

        return actions
