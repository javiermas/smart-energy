import numpy as np

from .base import Preprocessor


class BasicPreprocessor(Preprocessor):

    vars_to_float = ['energy_consumption_i', 'energy_generation_i', 'energy_excess_i', 'temperature',
                     'energy_to_battery_i', 'energy_to_grid_i', 'price_sell', 'price_buy'
                     #'energy_consumption_v', 'energy_generation_v', 'energy_to_battery_v', 'energy_to_grid_v',
                     ]
    vars_with_outliers = ['energy_consumption_i', 'energy_generation_i',
                          'energy_to_grid_i', 'energy_to_battery_i']
    vars_needing_offset = ['energy_consumption_i', 'energy_generation_i']
    #vars_to_wats = ['energy_consumption', 'energy_production', 'energy_excess',
    #                'energy_to_battery', 'energy_to_grid']

    def transform(self, data):
        _data = data['MinuteMeasurements'].copy()
        _data = self.rename_columns(_data)
        _data['solbox_id'] = _data['solbox_id'].astype(str)
        _data = _data.set_index(['solbox_id', 'timestamp'])
        _data_subset_float = self.strings_to_floats(_data[self.vars_to_float])
        _data = _data.drop(self.vars_to_float, axis=1).join(_data_subset_float)

        _data_without_outliers = self.remove_outliers(_data[self.vars_with_outliers], [5, 95])
        _data = _data.drop(self.vars_with_outliers, axis=1).join(_data_without_outliers)

        _data_with_offset = self.add_offset(_data[self.vars_needing_offset])
        _data = _data.drop(self.vars_needing_offset, axis=1).join(_data_with_offset)
        #_data = self.add_watts(_data)
        other_cols = ['energy_bought', 'energy_sold', 'battery_state_discrete']
        cols_to_keep = self.vars_to_float + other_cols
        data['MinuteMeasurements'] = _data[cols_to_keep].reset_index()
        return data

    #def add_watts(self, data):
    #    for var in self.vars_to_wats:
    #        if var in ['energy_excess', 'energy_generation']:
    #            data[var+'_w'] = data[var+'_i']

    #        data[var+'_w'] = self.compute_w_per_hour_for_two_minutes_data(data[var+'_i'], data[var+'_v'])

    #    return data

    @staticmethod
    def rename_columns(data):
        data = data.rename(columns={
            'TIME_DHAKA': 'timestamp',
            'fILoadDirect_avg': 'energy_consumption_i',
            'fIPV_avg': 'energy_generation_i',
            'fIExcess_avg': 'energy_excess_i',
            'fIToBat_avg': 'energy_to_battery_i',
            'fIToGrid_avg': 'energy_to_grid_i',
            'fPLoad_direct_avg': 'energy_consumption_v',
            'fVPV_avg': 'energy_generation_v',
            'fVExcess_avg': 'energy_excess_v',
            'fVBattery_avg': 'energy_to_battery_v',
            'fVGrid_avg': 'energy_to_grid_v',
            'fTemperature_avg': 'temperature',
            'fSellPrice': 'price_sell',
            'fBuyPrice': 'price_buy',
            'u8StateOfBattery': 'battery_state_discrete',
            'u32TotalBought': 'energy_bought',
            'u32TotalSold': 'energy_sold',
        })
        return data

    @staticmethod
    def strings_to_floats(data):
        assert all([t in [object, str] for t in data.dtypes])
        for v in data:
            data[v] = data[v].astype(str).apply(
                lambda x: x.replace(',', '.')).astype(float)

        return data

    def remove_outliers(self, data, percentiles=[5, 95]):
        for v in data:
            data[v] = data.groupby(['solbox_id'])[v].apply(
                self.remove_outliers_lambda, percentiles)

        return data

    @staticmethod
    def remove_outliers_lambda(data, percentiles):
        p0, p1 = np.percentile(data, percentiles)
        data.loc[(data > (p1 + 2 * (p1 - p0))) |
                 (data < (p0 - 2 * (p1 - p0)))] = np.nan
        return data

    @staticmethod
    def add_offset(data):
        for station in data.index.get_level_values(0).unique():
            for var in data:
                if data.loc[data.index.get_level_values(0) == station, var].min() < 0:
                    data.loc[data.index.get_level_values(0) == station, var] += np.abs(
                        data.loc[data.index.get_level_values(0) == station, var].min())

        return data

    @staticmethod
    def compute_w_per_hour_for_two_minutes_data(volt_var, current_var):
        return volt_var * current_var * 2 / 60

    @property
    def schema_input(self):
        return object
