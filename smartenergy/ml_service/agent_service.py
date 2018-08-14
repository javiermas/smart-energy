from .base import Service


class AgentService(Service):

    def __init__(self, agent):
        self.agent = agent
        self.action_space = self.agent.action_space

    def __call__(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)

    def train(self):
        pass

    def get_action(self, state):
        return self.agent.get_action(state)
