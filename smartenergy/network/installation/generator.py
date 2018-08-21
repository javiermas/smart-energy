from ...database import SimulatedMeasurements
from .base import InstallationElement


class Generator(InstallationElement):

    def __init__(self, pipes, connection=SimulatedMeasurements()):
        super().__init__(connection)
        self.pipes = pipes
        self.energy_supply = 0

    def initialize(self):
        pass

    def get_reading(self):
        return self.connection.get_last_generator_measurement(self.installation)

    def interact(self, action):
        self.energy_supply = action

    @property
    def action_space(self):
        return []
