import sys
import re
import logging

from datetime import timedelta
from pandas import DataFrame, concat

from .base import Service
from ..database import SimulatedMeasurements

TRUNCATE_BACKLOADING = timedelta(hours=50)

logging.basicConfig(stream=sys.stdout,
                    level='DEBUG',
                    format='%(levelname)s:%(asctime)s:%(name)s:::%(message)s')


class MLService(Service):

    def __init__(self, feature_service, forecast_service,
                 agent_service, repo=SimulatedMeasurements()):
        self.feature_service = feature_service
        self.forecast_service = forecast_service
        self.agent_service = agent_service
        self.action_space = agent_service.action_space
        self.repo = repo
        self.last_state = None

    def __call__(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)

    def initialize(self):
        self.agent_service.initialize()

    def train(self):
        logging.info('Training ML Service')
        data = self.repo.load()
        features = self.feature_service.get_features(data)
        self.forecast_service.train(features)
        #self.agent_service.train()

    def get_action(self, readings, random=False):
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
        actions = self.agent_service(state.reset_index(drop=True), random)
        return actions

    def feed_reward(self, reward):
        self.agent_service.feed_reward(reward)

    def _predictions_to_data(self, predictions):
        f = lambda x: 'prediction_' + str(int(x.group(0).split('_')[1]) + 1)
        predictions = [p.rename(columns={col: re.sub('(lag_)(\d)', f, col)
                       for col in p.columns}) for p in predictions.values()]
        return concat(predictions, axis=1)
