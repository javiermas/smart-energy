import sys
import re
import logging
from os import environ

from datetime import timedelta
from pandas import DataFrame, concat
from numpy.random import uniform

from .base import Service
from ..database import SimulatedMeasurements, Memories


TRUNCATE_BACKLOADING = timedelta(hours=50)

logging.basicConfig(stream=sys.stdout,
                    level='DEBUG',
                    format='%(levelname)s:%(asctime)s:%(name)s:::%(message)s')


class MLService(Service):

    num_memories = 10

    def __init__(self, feature_service, forecast_service,
                 agent_service, repo=SimulatedMeasurements()):
        self.feature_service = feature_service
        self.forecast_service = forecast_service
        self.agent_service = agent_service
        self.action_space = agent_service.action_space
        self.repo = repo
        self.memories_repo = Memories()
        self.last_state = None

    def __call__(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)

    def initialize(self):
        self.memories_repo.drop()

    def train(self):
        logging.info('Training ML Service')
        data = self.repo.load()
        features = self.feature_service.get_features(data)
        if environ['DEBUG'] != 'NETWORK':
            self.forecast_service.train(features)

        memories = self.memories_repo.load_n_random_memories(self.num_memories)
        self.agent_service.train(memories)

    def get_action(self, readings, random=True):
        data = self.repo.load_from(readings['datetime'].iloc[0] - TRUNCATE_BACKLOADING)
        data = concat([data, readings], sort=True)
        data = data.fillna(0)
        # Assumption: will only want to use the forecast for the next t
        features = DataFrame(self.feature_service.get_features(data).loc[readings['datetime'].iloc[[0]]])
        # TODO: implement forecasting as part of state
        #predictions = self.forecast_service(features)
        #predictions = self._predictions_to_data(predictions)
        #state = concat([predictions, features], axis=1)
        state = features
        if not random:
            actions = self.agent_service(state.reset_index(drop=True))
        else:
            actions = self.get_random_action()

        self.last_actions = actions
        self.last_state = state
        return actions

    def get_random_action(self):
        actions = {}
        for name, sub_space in self.action_space.items():
            actions[name] = {}
            for sub_space_name, sub_space_value in sub_space.items():
                actions[name][sub_space_name] = uniform(low=min(sub_space_value), high=max(sub_space_value))

        return actions

    def feed_reward(self, reward):
        self.store_memory(reward)

    def store_memory(self, reward):
        memory = {
            'state': self.last_state.reset_index(drop=True).squeeze().to_dict(),
            'actions': self.last_actions,
            'reward': reward,
        }
        self.memories_repo.insert_one(memory)

    def _predictions_to_data(self, predictions):
        f = lambda x: 'prediction_' + str(int(x.group(0).split('_')[1]) + 1)
        predictions = [p.rename(columns={col: re.sub('(lag_)(\d)', f, col)
                       for col in p.columns}) for p in predictions.values()]
        return concat(predictions, axis=1)
