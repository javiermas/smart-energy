from .base import Service
from ..database import Memories


class AgentService(Service):

    num_memories = 10

    def __init__(self, agent):
        self.agent = agent
        self.action_space = self.agent.action_space
        self.memories_repo = Memories()

    def __call__(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)

    def initialize(self):
        self.memories_repo.drop()

    def train(self):
        memories = self.memories_repo.load_n_random_memories(self.num_memories)
        self.agent.train(memories)

    def get_action(self, state, random):
        return self.agent.get_action(state, random)

    def feed_reward(self, reward):
        self.store_memory(reward)

    def store_memory(self, reward):
        memory = {
            'state': self.agent.last_state.reset_index(drop=True).squeeze().to_dict(),
            #'actions': self.agent.last_actions,
            'reward': reward,
        }
        self.memories_repo.insert_one(memory)

    def get_cumulative_reward(self, state, reward):
        self.agent.get_state_value(state)
