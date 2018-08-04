import tensorflow as tf
from .base import Network


class DeepNetwork(Network):

    def __init__(self, action_space: dict, num_features: int, num_actions: int, hidden_units: list):
        self.action_space = action_space
        self.num_features = num_features
        self.num_actions = num_actions
        self.hidden_units = hidden_units
        self.num_layers = len(self.hidden_units)

    def initialize(self):
        graph = tf.Graph()
        with graph.as_default():
            state = tf.placeholder(tf.float32, [None, self.num_features], name='state')
            reward = tf.placeholder(tf.float32, None, name='reward')
            action = tf.placeholder(tf.int32, [None], name='action')
            next_state = tf.placeholder(tf.float32, [None, self.num_features], name='next_state')

            self.parameters = self._create_parameters(state, action)
            self.layers = self._create_layers(state, action)
            optimal_action = tf.argmax(self.layers['layer_out'], axis=1)
            Q_target = tf.placeholder(shape=[None], dtype=tf.float32)

            self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)
            
            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
            
            self.td_error = tf.square(self.targetQ - self.Q)
            self.loss = tf.reduce_mean(self.td_error)
            self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.updateModel = self.trainer.minimize(self.loss)

 

    def _create_parameters(self, _input, _output):
        parameters = {
            'weights_0': self.create_variable([_input.get_shape()[1], self.hidden_units[0]], 'weights_in'),
            'bias_0': self.create_variable([self.hidden_units[-1]], 'bias_in'),
            'weights_out': self.create_variable([self.hidden_units[-1], _output.shape()],
                                                                      'weights_out'),
            'bias_out': self.create_variable([_output.shape], 'bias_out')
        }
        if len(self.hidden_units) > 1:
            for i, units in enumerate(self.hidden_units):
                parameters[f'weight_{i+1}'] = self.create_variable(
                    [parameters[f'weight_{i}'].get_shape()[1], units], f'weights_{i+1}')
                parameters[f'bias_{i+1}'] = self.create_variable([units], f'bias_{i+1}')

        return parameters
    
    def _create_layers(self, _input, _output):
        layers = {
            'layer_0': self.create_layer(
                tf.matmul(_input, self.parameters['weights_0']) + self.parameters['bias_0']),
            'layer_out': self.create_layer(
                tf.matmul(_input, self.parameters['weights_out']) + self.parameters['bias_out']),
        }
        if len(self.hidden_units) > 1:
            for i, units in enumerate(self.hidden_units):
                layers[f'layer_{i+1}'] = self.create_layer(
                    tf.matmul(layers[f'layer_{i}'], self.parameters[f'weights_{i}']), +
                    self.parameters[f'bias_{i}'])

        return layers

    @staticmethod
    def create_variable(size, name=None):
        name = name or str(size)
        return tf.Variable(tf.random_uniform(size, 0, 0.01), name=name)

    @staticmethod
    def create_layer(features, activation=tf.nn.relu):
        return activation(features)

    def chain(length):
        return ((i, i+1) for i in range(length))
