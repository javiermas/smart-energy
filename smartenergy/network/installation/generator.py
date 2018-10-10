from .base import InstallationElement


class Generator(InstallationElement):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy_supply = 0

    def initialize(self):
        pass

    def get_reading(self):
        return self.data_stream.get_last_generator_measurement(self.installation)

    def interact(self, action):
        self.energy_supply = action

    @property
    def action_space(self):
        return []
