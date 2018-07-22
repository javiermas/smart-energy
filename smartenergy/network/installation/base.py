from abc import abstractmethod
from ..base import NetworkElement


class Installation(NetworkElement):

    def __init__(self, identifier, elements, grid):
        super().__init__()
        self.id = identifier
        self.elements = elements
        self.grid = grid

    def __repr__(self):
        return f'installation_{self.id}'

    def get_reading(self, t):
        return {elem.__class__.__name__: elem.get_reading(t) for elem in self.elements}

    def interact(self, actions):
        for elem in self.elements:
            elem.interact(actions[elem.__class__.__name__])


class InstallationElement(NetworkElement):

    def __init__(self, connection):
        super().__init__()
        self.connection = connection

    @abstractmethod
    def get_reading(self):
        pass

    @property
    def action_space(self):
        return []
