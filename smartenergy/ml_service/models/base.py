from abc import ABC, abstractmethod

from ...logger import Logger


class Model(ABC):

    def __init__(self):
        self.log = Logger(self.__class__.__name__)

    @abstractmethod
    def train(self):
        pass
