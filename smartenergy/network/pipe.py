from numpy.random import normal
from .base import NetworkElement


class Pipe(NetworkElement):

    def __init__(self, resistance):
        super().__init__()
        self.resistance = resistance

    def apply_resistance(self, current):
        return min(0, current * normal(self.resistance, 0.1))

    def get_reading(self):
        pass

    def interact(self):
        pass
