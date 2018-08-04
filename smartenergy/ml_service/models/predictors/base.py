import logging
from pandas import DataFrame, Series
from abc import ABC, abstractmethod


class Predictor(ABC):

    def __init__(self, hyperparameters, id_kwargs=None):
        self.hyperparameters = hyperparameters
        self.id_kwargs = id_kwargs or {}

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @staticmethod
    def apply_schemata(func):
        def new_func(self, features=None, target=None):
            kwargs = {}
            if features is not None:
                keys = features if isinstance(features, DataFrame) else features.keys()
                feature_names = [f for f in keys if f in self.feature_schema.keys()]
                kwargs['features'] = features[feature_names]

            if target is not None:
                kwargs['target'] = target[[*self.target_schema.keys()]]

            return func(self, **kwargs)

        return new_func

    @property
    @abstractmethod
    def feature_schema(self):
        pass

    @property
    @abstractmethod
    def target_schema(self):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def train(self, data, target):
        pass

    @abstractmethod
    def validate(self, data, target):
        pass

    @abstractmethod
    def serialize(self, stream):
        pass

    @abstractmethod
    def unserialize(self, stream):
        pass
