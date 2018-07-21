from pandas import DataFrame, concat
from .measurements import Measurements
from .connection import Connection


class MinuteMeasurements(Measurements):

    def __init__(self, connection=Connection.get_connection()):
        super().__init__('minute_measurements', connection)
