import sys
import re

from datetime import timedelta
from pandas import DataFrame, concat

from .base import Service
from ..database import Memories

TRUNCATE_BACKLOADING = timedelta(hours=50)


class MLService(Service):

    num_memories = 10

    def __init__(self, feature_service, forecast_service,
                 agent_service, data_stream):
        super().__init__()
        self.feature_service = feature_service
        self.forecast_service = forecast_service
        self.agent_service = agent_service
        self.action_space = agent_service.action_space
        self.data_stream = data_stream
        self.memories_repo = Memories()

    def serve(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)

    def initialize(self):
        self.memories_repo.drop()

    def train(self):
        self.log.info('Training ML Service')
        features = self.feature_service(self.data_stream.get_all_measurements())
        forecast_loss = self.forecast_service.train(features)
        memories = self.memories_repo.load_n_random_memories(self.num_memories)
        agent_loss = self.agent_service.train(memories)
        return {'forecast_service': forecast_loss, 'agent_service': agent_loss}

    def get_action(self, readings, random=False):
        state = self.create_state(readings)
        actions = self.agent_service(state, random)
        return actions

    def get_state_value(self, readings):
        state = self.create_state(readings)
        return self.agent_service.get_state_value(state)

    def create_state(self, readings):
        t = readings.index.get_level_values('datetime')[0]
        data = self.data_stream.get_all_measurements_since(t - TRUNCATE_BACKLOADING)
        data = concat([data, readings], sort=True)
        data = data.fillna(0)
        # Assumption: will only want to use the forecast for the next t
        features = DataFrame(self.feature_service(data.reset_index()).loc[[t]])
        # TODO: implement forecasting as part of state
        #predictions = self.forecast_service(features)
        #predictions = self._predictions_to_data(predictions)
        #state = concat([predictions, features], axis=1)
        state = features
        return state.reset_index(drop=True)

    def feed_reward(self, reward):
        self.store_memory(reward)

    def _predictions_to_data(self, predictions):
        f = lambda x: 'prediction_' + str(int(x.group(0).split('_')[1]) + 1)
        predictions = [p.rename(columns={col: re.sub('(lag_)(\d)', f, col)
                       for col in p.columns}) for p in predictions.values()]
        return concat(predictions, axis=1)

    def store_memory(self, reward):
        memory = {
            'state': self.agent_service.agent.last_state.reset_index(drop=True).squeeze().to_dict(),
            'actions': self.agent_service.agent.last_actions,
            'reward': reward,
        }
        self.memories_repo.insert_one(memory)
