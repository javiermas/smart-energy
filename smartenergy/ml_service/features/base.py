from abc import ABC, abstractmethod
from ...exceptions import SchemaException

class Transformer(ABC):

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    @abstractmethod
    def transform(self, data):
        pass

    @property
    @abstractmethod
    def schema(self):
        pass

    @staticmethod
    def apply_schemata(func):
        def new_func(self, data):
            schema = self.schema
            data_needed = data.copy()
            if data.empty:
                raise SchemaException(f'DataFrame passed to {self.__class__.__name__} was empty')
            elif any(c not in data.columns for c in schema):
                missing_column = next(c for c in schema if c not in data.columns)
                raise SchemaException(f'DataFrame passed to {self.__class.__name__} missing the {missing_column} column')
            else:
                data_needed = data[[*schema.keys()]]

            return func(self, data_needed)

        return new_func


class Feature(Transformer):

    @abstractmethod
    def transform(self, data):
        pass

    @property
    @abstractmethod
    def schema(self):
        pass


class Preprocessor(Transformer):

    @abstractmethod
    def transform(self, data):
        pass

    @property
    @abstractmethod
    def schema(self):
        pass

