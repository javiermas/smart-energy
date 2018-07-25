import sys
import logging
from pandas import concat
from datetime import timedelta
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

    def __call__(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)

    def train(self):
        logging.info('Training ML Service')
        data = self.repo.load()
        features = self.feature_service.get_features(data)
        self.forecast_service.train(features)
        self.agent_service.train()

    def get_action(self, readings):
        data = self.repo.load_from(readings['datetime'].iloc[0] - TRUNCATE_BACKLOADING)
        data = concat([data, readings])
        features = self.feature_service.get_features(data)
        predictions = self.forecast_service(features)
        actions = self.agent_service(readings, features, predictions)
        actions = {i_name: {e: 1 for e in i.keys()} for i_name, i in self.action_space.items()}
        self.store_readings(readings)
        return actions

    def store_readings(self, readings):
        self.repo.insert_many(readings)
    
