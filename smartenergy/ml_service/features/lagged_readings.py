from numbers import Number
from pandas import pivot_table

from .base import Feature


class LaggedReadings(Feature):

    def __init__(self, lags):
        super().__init__()
        self.lags = lags
        self.features_to_lag = ['energy_generation_i', 'energy_consumption_i', 'energy_to_grid_i',
                                'energy_to_battery_i', 'temperature', 'battery_state_discrete']
        self.lagged_features = [f'{f}_lag_{lag}' for f in self.features_to_lag
                                for lag in range(0, self.lags + 1)]

    def transform(self, data):
        features = data.copy()
        features = self.lag_features(features)
        features = self.pivot_data(features)
        return features

    def lag_features(self, data):
        for lag in range(self.lags + 1):
            colnames = [f'{feature}_lag_{lag}' for feature in self.features_to_lag]
            data[colnames] = data.groupby('solbox_id')[self.features_to_lag].shift(lag).fillna(0)

        return data

    def pivot_data(self, data):
        data = data.set_index(['solbox_id', 'datetime'])[self.lagged_features+self.features_to_lag]
        data = pivot_table(data, columns='solbox_id', index='datetime')
        data.columns = ['_station'.join(col).strip() for col in data.columns.values]
        return data
    
    @property
    def schema_input(self):
        schema_core = {
            'solbox_id': int,
            'datetime': object,
        }
        schema_features = {f: Number for f in self.features_to_lag}
        return {**schema_core, **schema_features}
