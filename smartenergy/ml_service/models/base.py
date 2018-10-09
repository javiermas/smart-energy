from abc import ABC, abstractmethod

from ...logger import Logger


class Model(ABC):

    def __init__(self):
        self.log = Logger()

    @abstractmethod
    def train(self):
        pass
