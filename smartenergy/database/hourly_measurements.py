from .measurements import Measurements
from .connection import Connection


class HourlyMeasurements(Measurements):

    def __init__(self, connection=Connection.get_connection()):
        super().__init__('hourly_measurements', connection)
