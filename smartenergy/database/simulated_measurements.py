from .measurements import Measurements
from .connection import Connection


class SimulatedMeasurements(Measurements):

    def __init__(self, connection=Connection.get_connection()):
        super().__init__('simulated_measurements', connection)
