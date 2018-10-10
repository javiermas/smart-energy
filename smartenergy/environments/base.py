from abc import abstractmethod, ABC

from .metrics_manager import MetricsManager
from ..logger import Logger



class Environment(ABC):

    def __init__(self, ml_service, network, data_stream, burning_steps, init_steps, step_size):
        self.log = Logger(self.__class__.__name__)
        self.network = network
        self.ml_service = ml_service
        self.data_stream = data_stream
        self.burning_steps = burning_steps
        self.init_steps = init_steps
        self.step_size = step_size
        self.metrics_manager = MetricsManager(self.data_stream)
    
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
