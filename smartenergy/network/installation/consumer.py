from .base import InstallationElement
from ...database import DataStream


class Consumer(InstallationElement):

    def __init__(self, data_stream):
        self.data_stream = data_stream
    
    def initialize(self):
        pass

    def get_reading(self):
        return self.data_stream.get_last_consumer_measurement(self.installation)

    def interact(self, action):
        pass
