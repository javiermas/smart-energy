import logging
from abc import abstractmethod, ABC
from ..database import HourlyMeasurements, SimulatedMeasurements


class Environment(ABC):

    def __init__(self, ml_service, network, init_t, init_steps, step_size,
                 source_repo=HourlyMeasurements(), mirror_repo=SimulatedMeasurements()):
        self.network = network
        self.ml_service = ml_service
        self.init_t = init_t
        self.init_steps = init_steps
        self.t = self.init_t
        self.step_size = step_size
        self.source_repo = source_repo
        self.mirror_repo = mirror_repo
    
    @abstractmethod
    def run(self, steps):
        pass
    
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def step(self):
        pass
