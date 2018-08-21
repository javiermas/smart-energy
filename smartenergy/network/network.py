from .base import NetworkElement


class Network(NetworkElement):

    def __init__(self, installations=None):
        super().__init__()
        self.installations = installations or {}

    def initialize(self):
        for installation in self.installations.values():
            installation.initialize()

    def update(self):
        for installation in self.installations.values():
            installation.update()

    def get_reading(self):
        return {class_name: installation.get_reading() for class_name, installation in self.installations.items()}

    def interact(self, actions):
        for _id, installation in self.installations.items():
            installation.interact(actions[_id])

    def add_installation(self, installation):
        self.installations[str(installation)] = installation

    @property
    def action_space(self):
        action_space = {
            str(installation): {
                elem.__class__.__name__: elem.action_space for elem in installation.elements
            } for installation in self.installations.values()
        }
        return action_space
