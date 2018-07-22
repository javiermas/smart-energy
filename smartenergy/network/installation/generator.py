from ...database import HourlyMeasurements
from .base import InstallationElement


class Generator(InstallationElement):

    def __init__(self, pipes, connection=HourlyMeasurements()):
        super().__init__(connection)
        self.pipes = pipes

    def get_reading(self, t):
        return self.connection.get_generator_measurement(t)

    def interact(self, action):
        pass

    @property
    def action_space(self):
        return []
