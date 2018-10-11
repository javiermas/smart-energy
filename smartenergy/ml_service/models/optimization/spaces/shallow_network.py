from random import choice
from numpy import geomspace


def shallow_network_space():
    space = {
        'learning_rate': choice(geomspace(1e-3, 1e-1)),
        'hidden_units': choice(range(1, 10)),
    }
    return space
