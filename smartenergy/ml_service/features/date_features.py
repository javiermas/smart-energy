from pandas import pivot_table
from .base import Feature


class DateFeatures(Feature):

    def __init__(self, lags):
        super().__init__()

    @Feature.apply_schemata
    def transform(self, data):
        features = data['SimulatedMeasurements'].copy()
        features = self.lag_features(features)
        features = self.pivot_data(features)
        return features

    def get_date_features(self, data):
        return data

    def pivot_data(self, data):
        data = pivot_table(data[self.lagged_features+self.features_to_lag], 
                           columns='solbox_id', index='datetime')
        data.columns = ['_station'.join(col).strip() for col in data.columns.values]
        return data
