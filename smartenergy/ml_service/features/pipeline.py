import logging

from .base import Transformer


class Pipeline(Transformer):

    def __init__(self, preprocessors, features):
        super().__init__()
        self.preprocessors = preprocessors
        self.features = features

    def transform(self, data):
        for preprocessor in self.preprocessors:
            self.log.info(f"Running preprocessor {preprocessor.__class__.__name__}")
            data = preprocessor.transform(data)
        
        for feature in self.features:
            self.log.info(f"Running feature {feature.__class__.__name__}")
            data[feature.__class__.__name__] = feature.transform(data)

        return data

    def schema_input(self):
        return object
