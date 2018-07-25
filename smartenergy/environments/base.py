from abc import abstractmethod, ABC


class Environment(ABC):

    def __init__(self):
        pass

    def run(self, steps):
        for step in range(steps):
            self.step()

    @abstractmethod
    def step(self):
        pass


class SBEnvironment(Environment):

    def __init__(self, ml_service, network, init_t, step_size):
        super().__init__()
        self.network = network
        self.ml_service = ml_service
        self.t = init_t
        self.step_size = step_size

    def step(self):
        self.t += self.step_size
        readings = self.network.get_reading(self.t)
        actions = self.ml_service.get_action(readings)
        self.network.interact(actions)

    def get_excess_battery(self, readings):
        return 1
