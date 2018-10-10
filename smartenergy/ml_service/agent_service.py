from .base import Service


class AgentService(Service):

    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.action_space = self.agent.action_space

    def serve(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)

    def train(self, memories):
        self.log.info(f'Training agent {self.agent.__class__.__name__} with {len(memories)} memories')
        loss = self.agent.train(memories)
        self.log.info(f'Trained agent {self.agent.__class__.__name__}')
        return loss

    def get_action(self, state, random):
        return self.agent.get_action(state, random)

    def get_state_value(self, state):
        return self.agent.get_state_value(state)
