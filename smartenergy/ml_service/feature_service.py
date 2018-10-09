from functools import reduce
from pandas import merge

from .base import Service


class FeatureService(Service):

    def __init__(self, features: list):
        super().__init__()
        self.features = features

    def get_features(self, data):
        feature_data = dict()
        for feature in self.features:
            feature_data[feature.__class__.__name__] = feature(data)

        data = self.merge_features(feature_data).sort_index()
        return data

    def merge_features(self, data):
        data = reduce(lambda l, r: merge(l, r, left_index=True, right_index=True), data.values())
        return data
