from abc import abstractmethod

from ..base import Model


class Agent(Model):

    def __call__(self):
        self.train()

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def get_action(self):
        pass
    

class Network(object):
    pass
