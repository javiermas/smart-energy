from abc import abstractmethod, ABC

from ..logger import Logger


class Service(ABC):

    def __init__(self):
        self.log = Logger(self.__class__.__name__)
    
    def __call__(self, *args, **kwargs):
        return self.serve(*args, **kwargs)

    @abstractmethod
    def serve(self):
        pass
