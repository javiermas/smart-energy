from abc import abstractmethod, ABC

from .metrics_manager import MetricsManager
from ..logger import Logger
from ..database import HourlyMeasurements, SimulatedMeasurements



class Environment(ABC):

    def __init__(self, ml_service, network, burning_steps, init_steps, step_size,
                 source_repo=HourlyMeasurements(), mirror_repo=SimulatedMeasurements()):
        self.log = Logger()
        self.network = network
        self.ml_service = ml_service
        self.burning_steps = burning_steps
        self.init_steps = init_steps
        self.step_size = step_size
        self.source_repo = source_repo
        self.mirror_repo = mirror_repo
        self.metrics_manager = MetricsManager(self.source_repo)
    
    @abstractmethod
    def run(self, steps):
        pass
    
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def restart(self):
        pass
