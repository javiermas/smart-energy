import logging
from .base import Service
from ..database import Models


class ForecastService(Service):

    def __init__(self, models=None, connection=Models()):
        self.connection = connection
        self.models = models or self.connection.load()
        self.trained_models = {}

    def __call__(self, features):
        return self.get_forecast(features)

    def get_forecast(self, features):
        if not self.trained_models:
            raise ValueError('No trained models')

        predictions = {}
        for model in self.trained_models:
            predictions[str(model)] = model.predict(features)

        return predictions

    def train(self, data):
        for model in self.models:
            if list(model.target_schema.keys())[0] not in data.columns:
                logging.debug(f'Missing target for station {str(model)}')
                continue

            model_features = [col for col in model.feature_schema.keys() if col in data.columns]
            missing_features = [col for col in model.feature_schema.keys() if col not in data.columns]
            if missing_features:
                logging.debug(f'Missing features \n {missing_features}')

            features, target = data[model_features], data[[*model.target_schema.keys()]]
            model.train(features, target)
            self.trained_models[str(model)] = model
            logging.info('{} model stored (validation score: {})'
                         .format(model.__class__.__name__, model.validate(features, target)))
