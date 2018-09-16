from abc import ABC, abstractmethod


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
    def schema_input(self):
        pass


class Feature(Transformer):

    @abstractmethod
    def transform(self, data):
        pass

    @property
    @abstractmethod
    def schema_input(self):
        pass


class Preprocessor(Transformer):

    @abstractmethod
    def transform(self, data):
        pass

    @property
    @abstractmethod
    def schema_input(self):
        pass
