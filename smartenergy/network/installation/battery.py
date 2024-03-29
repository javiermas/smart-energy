from numpy.random import normal

from .base import InstallationElement


class Battery(InstallationElement):

    def __init__(self, data_stream, capacity):
        super().__init__(data_stream)
        self.state = None
        self.capacity = capacity

    def initialize(self):
        init_measurement = self.data_stream.get_last_battery_measurement(self.installation)
        if init_measurement is None:
            self.state = 50
            return

        self.state = abs(normal(init_measurement, 6))

    def get_reading(self):
        return self.state

    def interact(self, action):
        pass

    def update(self, update):
        update = update / self.capacity
        self.state = round(max(0, min(100, self.state + update)), 4)

    def initialize_state(self, t):
        self.connection.get_battery_measurement(t)

    @property
    def action_space(self):
        return []
