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
        for element in self.elements.values():
            element._assign_id(self.id)

    def __repr__(self):
        return f'installation_{self.id}'

    def initialize(self):
        for element in self.elements.values():
            element.initialize()

    def update(self):
        consumed_energy = self.elements['consumer'].get_reading()
        if consumed_energy is None:
            return

        energy_supplied_by_generator = consumed_energy * self.elements['generator'].energy_supply
        generated_energy = self.elements['generator'].get_reading()
        battery_update = generated_energy - energy_supplied_by_generator
        self.elements['battery'].update(battery_update)

    def get_reading(self):
        return {elem.__class__.__name__: elem.get_reading() for elem in self.elements.values()}

    def interact(self, actions):
        for element, action in actions.items():
            self.elements[element].interact(action)


class InstallationElement(NetworkElement):

    def __init__(self, data_stream):
        super().__init__()
        self.data_stream = data_stream

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
    
