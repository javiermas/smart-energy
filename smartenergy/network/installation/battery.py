from .base import InstallationElement
from ...database import HourlyMeasurements


class Battery(InstallationElement):

    def __init__(self, pipes, connection=HourlyMeasurements()):
        super().__init__(connection)
        self.pipes = pipes
        self.state = None

    def get_reading(self, t):
        return self.state or self.connection.get_battery_measurement(t)

    def interact(self, action):
        pass

    def update_state(self, update):
        self.state += update

    def initialize_state(self, t):
        self.connection.get_battery_measurement(t)

    @property
    def action_space(self):
        return []
