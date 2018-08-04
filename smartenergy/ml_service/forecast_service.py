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
        for model_name, model in self.trained_models.items():
            predictions[model_name] = model.predict(features)

        return predictions

    def train(self, data):
        for model in self.models:
            train_data = data.copy()
            if self._is_target_missing(model, train_data.columns):
                logging.debug(f'Missing target for {str(model)}')
                continue

            train_data = train_data.dropna(subset=[*model.target_schema])
            assert not train_data.empty, f'Target should be missing for {str(model)}'
            if len(train_data) < 24 * 7:
                logging.debug(f'Target too small ({len(data)}) for {str(model)}')
                continue

            model_features = [col for col in model.feature_schema.keys() if col in train_data.columns]
            features, target = train_data[model_features], train_data[[*model.target_schema]]
            model.train(features, target)
            self.trained_models[str(model)] = model
            loss = model.validate(features, target)
            logging.info(f'{str(model)} model stored (validation score: {loss})')

    @staticmethod
    def _is_target_missing(model, columns):
        return list(model.target_schema.keys())[0] not in columns
