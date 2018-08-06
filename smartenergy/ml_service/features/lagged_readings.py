from pandas import pivot_table
from .base import Feature


class LaggedReadings(Feature):

    def __init__(self, lags):
        super().__init__()
        self.lags = lags
        self.features_to_lag = ['fILoadDirect_avg', 'fILoad_avg', 'fIPV_avg',
                                'fIExcess_avg', 'fIToGrid_avg', 'fIToBat_avg',
                                'fTemperature_avg', 'u8StateOfBattery']
        self.lagged_features = [f'{f}_lag_{lag}'.format(f, lag) for f in self.features_to_lag
                                for lag in range(0, self.lags + 1)]

    @Feature.apply_schemata
    def transform(self, data):
        features = data.copy()
        features = self.lag_features(features)
        features = self.pivot_data(features)
        return features

    def lag_features(self, data):
        for lag in range(self.lags + 1):
            colnames = [f'{f}_lag_{lag}' for f in self.features_to_lag]
            data[colnames] = data.groupby('solbox_id')[self.features_to_lag].shift(lag).fillna(0)

        return data

    def pivot_data(self, data):
        data = data.set_index(['solbox_id', 'datetime'])[self.lagged_features+self.features_to_lag]
        data = pivot_table(data, columns='solbox_id', index='datetime')
        data.columns = ['_station'.join(col).strip() for col in data.columns.values]
        return data
    
    @property
    def schema(self):
        schema = {
            'solbox_id': int,
            'datetime': object,
            'fILoadDirect_avg': float,
            'fILoad_avg': float,
            'fIPV_avg': float,
            'fIExcess_avg': float,
            'fIToGrid_avg': float,
            'fIToBat_avg': float,
            'fTemperature_avg': float,
            'u8StateOfBattery': int,
        }
        return schema
