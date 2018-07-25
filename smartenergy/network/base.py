from abc import abstractmethod, ABC


class Environment(ABC):

    def __init__(self):
        pass


class NetworkElement(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_reading(self, t):
        pass

    @abstractmethod
    def interact(self, actions):
        pass
