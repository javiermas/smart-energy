from .base import InstallationElement
from ...database import HourlyMeasurements


class Consumer(InstallationElement):

    def __init__(self, connection=HourlyMeasurements()):
        super().__init__(connection)

    def get_reading(self, t):
        return self.connection.get_consumer_measurement(t)

    def interact(self, action):
        pass
