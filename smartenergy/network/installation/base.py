from abc import abstractmethod
from ..base import NetworkElement


class Installation(NetworkElement):

    def __init__(self, identifier, elements, grid):
        super().__init__()
        self.id = identifier
        self.elements = elements
        self.grid = grid
        self.assign_ids_to_elements()

    def assign_ids_to_elements(self):
        for element in self.elements:
            element._assign_id(self.id)

    def __repr__(self):
        return f'installation_{self.id}'

    def initialize(self):
        for element in self.elements:
            element.initialize()

    def get_reading(self):
        return {elem.__class__.__name__: elem.get_reading() for elem in self.elements}
        

    def interact(self, actions):
        for elem in self.elements:
            elem.interact(actions[elem.__class__.__name__])


class InstallationElement(NetworkElement):

    def __init__(self, connection):
        super().__init__()
        self.connection = connection

    def _assign_id(self, _id):
        self.installation = _id

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def get_reading(self):
        pass

    @property
    def action_space(self):
        return []
    
