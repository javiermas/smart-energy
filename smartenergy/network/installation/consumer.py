from .base import InstallationElement
from ...database import SimulatedMeasurements


class Consumer(InstallationElement):

    def __init__(self, connection=SimulatedMeasurements()):
        super().__init__(connection)
    
    def initialize(self):
        pass

    def get_reading(self):
        return self.connection.get_last_consumer_measurement(self.installation)

    def interact(self, action):
        pass
