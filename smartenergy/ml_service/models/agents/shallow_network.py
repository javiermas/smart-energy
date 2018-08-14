import tensorflow as tf
from .base import Network


class ShallowNetwork(Network):

    def __init__(self, action_space: dict, hidden_units: int, learning_rate: float):
        self.action_space = action_space
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.num_actions = sum([len(elem_values) for key, value in action_space.items()
                                for elem_keys, elem_values in value.items()])
        self.initializer = tf.random_normal

    def initialize(self):
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.state = tf.placeholder(tf.float32, [1, None], name='state')
            #reward = tf.placeholder(tf.float32, None, name='reward')
            #next_state = tf.placeholder(tf.float32, [None, self.state_size], name='next_state')
            # Replace Q by reward 
            W = tf.Variable(
                self.initializer(shape=[tf.shape(self.state)[1], self.hidden_units * self.num_actions]),
                name='common_weights',
                validate_shape=False)
            b = tf.Variable(self.initializer(shape=[self.hidden_units * self.num_actions]),
                            name='common_bias', validate_shape=False)
            common_layer = tf.nn.relu(tf.matmul(self.state, W) + b, name='common_layer')
            weights, Q_out, Q_next, actions, actions_taken, losses = [{}] * 6
            for key, sub_space in self.action_space.items():
                weights[key], Q_out[key], Q_next[key], actions[key], actions_taken[key], losses[key] = [{}] * 6
                for sub_space_key, sub_space_value in sub_space.items():
                    Q_next[key][sub_space_key] = tf.placeholder(
                        tf.float32, [1, len(self.action_space[key][sub_space_key])],
                        name=f'Q_next_{key}_{sub_space_key}')
                    weights[key][sub_space_key] = tf.Variable(
                        self.initializer(shape=[tf.shape(common_layer)[1], max(sub_space_value)]),
                        name=f'weights_{key}_{sub_space_key}',
                        validate_shape=False)
                    _Q_out = tf.matmul(common_layer, weights[key][sub_space_key])
                    actions[key][sub_space_key] = tf.argmax(_Q_out)
                    Q_out[key][sub_space_key] = _Q_out
                    losses[key][sub_space_key] = tf.losses.mean_squared_error(
                        Q_next[key][sub_space_key], Q_out[key][sub_space_key])

            self.loss = tf.reduce_mean([loss for sub_space_losses in losses.values()
                                  for loss in sub_space_losses.values()])
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            tf.summary.FileWriter("./logs/agraph", self.graph).close()
            update = optimizer.minimize(self.loss)
            self.actions = actions
            self.reward = Q_out
