import tensorflow as tf
from .base import Network


class ShallowNetwork(Network):

    def __init__(self, action_space: dict, state_size: int, learning_rate: float):
        self.action_space = action_space
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.num_actions = sum([len(key.items()) for key in action_space.items()])

    def initialize(self):
        graph = tf.Graph()
        with graph.as_default():
            state = tf.placeholder(tf.float32, [None, self.state_size], name='state')
            #reward = tf.placeholder(tf.float32, None, name='reward')
            #next_state = tf.placeholder(tf.float32, [None, self.state_size], name='next_state')
            # Replace Q by reward 
            W = tf.Variable(tf.random_uniform([self.state_size, self.num_actions], 0, 0.01))
            weights, Q_out, Q_next, actions, actions_taken, losses = [{}] * 6
            for key, sub_space in self.action_space.items():
                weights[key], Q_out[key], Q_next[key], actions[key], actions_taken[key], losses[key] = [{}] * 6
                for sub_space_key, sub_space_value in sub_space.items():
                    weights[key][sub_space_key] = tf.Variable(
                        tf.random_uniform([W.get_shape()[1], sub_space_value], 0, 0.01)),
                    Q_out[key][sub_space_key] = tf.matmul(state, weights[key][sub_space_key])
                    actions[key][sub_space_key] = tf.argmax(Q_out[key][sub_space_key])
                    Q_next[key][sub_space_key] = tf.placeholder(
                        tf.float32, [1, self.action_space[key][sub_space_key]])
                    losses[key][sub_space_key] = tf.reduce_sum(
                        tf.square(Q_next[key][sub_space_key] - Q_out[key][sub_space_key]))
             
            self.loss = tf.add_n([loss for sub_space_losses in losses.values()
                                  for loss in sub_space_losses.values()])
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            update = optimizer.minimize(self.loss)
