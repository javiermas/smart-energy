from abc import abstractmethod, ABC

from ..logger import Logger


class Service(ABC):

    def __init__(self):
        self.log = Logger()
