import logging
from os import getenv
from uuid import uuid4

from .base import Transformer


class Pipeline(Transformer):

    def __init__(self, preprocessors, features):
        self.preprocessors = preprocessors
        self.features = features
        self._set_up_logging()

    def transform(self, data):
        for preprocessor in self.preprocessors:
            logging.info(f"Running preprocessor {preprocessor.__class__.__name__}")
            data = preprocessor.transform(data)
        
        for feature in self.features:
            logging.info(f"Running feature {feature.__class__.__name__}")
            data[feature.__class__.__name__] = feature.transform(data)

        return data

    def _set_up_logging(self):
        class_name = self.__class__.__name__
        self.log = logging.getLogger(class_name + str(uuid4()))
        self.log.setLevel(getenv('{}_LOG_LEVEL'.format(class_name.upper()), 'INFO'))
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'%(levelname)s:::{class_name}:::%(asctime)s:::%(message)s')
        handler.setFormatter(formatter)
        self.log.addHandler(handler)

    def schema_input(self):
        return object
