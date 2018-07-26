from .base import Service


class AgentService(Service):

    def __init__(self):
        self.action_space = None

    def train(self):
        pass

    def get_action(self):
        pass
