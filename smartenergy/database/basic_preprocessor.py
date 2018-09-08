import datetime
import numpy as np
import pandas as pd
from ..ml_service.features.base import Preprocessor


class BasicPreprocessor(Preprocessor):

    discrete_to_percent = {0: 0., 1: 25., 2: 50., 3: 75., 4: 100.}
    battery_capacities = {  # w_h
        "30": 371,
        "32": 204,
        "34": 166,
        "37": 115,
        "40": 303,
        "43": 136,
        "47": 150,
        "48": 205,
        "68": 194,
        "71": 402,
        "97": 597,
        "190": 67,
        "309": 199,
        "344": 71,
        "345": 504,
        "366": 340,
        "369": 77,
    }

    def transform(self, data):
        data = self.rename_columns(data)
        data = self.add_date_features(data)
        data['solbox_id'] = data['solbox_id'].astype(str)
        data = data.set_index(['solbox_id', 'datetime']).sort_index()

        vars_to_float = ['energy_consumption_i', 'energy_production_i', 'energy_excess_i',
                         'energy_to_battery_i', 'energy_to_grid_i', 'temperature',
                         'price_sell', 'price_buy']
        data_subset_float = self.strings_to_floats(data[vars_to_float])
        data = data.drop(vars_to_float, axis=1).join(data_subset_float)

        vars_with_outliers = ['energy_consumption_i', 'energy_production_i',
                              'energy_to_grid_i', 'energy_to_battery_i']
        data_without_outliers = self.remove_outliers(data[vars_with_outliers], [5, 95])
        data = data.drop(vars_with_outliers, axis=1).join(data_without_outliers)

        vars_needing_offset = ['energy_consumption_i', 'energy_production_i']
        data_with_offset = self.add_offset(data[vars_needing_offset])
        data = data.drop(vars_needing_offset, axis=1).join(data_with_offset)
        '''
        data = self.add_battery_state_features(data)
        data['energy_production_computed_i'] = self.compute_energy_production(data)
        data['energy_consumption_computed_i'] = self.compute_energy_consumption(data)
        '''
        date_cols = ['year', 'month', 'week', 'weekday', 'day', 'hour']
        '''
        battery_cols = ['battery_state_continuous_at_change', 'battery_state_continuous_theoretical',
                        'battery_state_continuous_computed', 'energy_to_battery_untracked_i']
        '''
        other_cols = ['energy_bought', 'energy_sold', 'battery_state_discrete']
          #            'energy_production_computed_i', 'energy_consumption_computed_i']
        cols_to_keep = vars_to_float + date_cols + other_cols
        return data[cols_to_keep].sort_index()

    def add_battery_state_features(self, data):
        data['battery_state_change'] = self.compute_battery_state_change(data['battery_state_discrete'])
        data = self.add_battery_state_mappings(data)
        data = self.drop_data_before_battery_change(data)
        data['battery_state_continuous_theoretical'] = self.compute_battery_state_theoretical(data)
        data = self.add_energy_to_battery_untracked(data)
        return data

    @staticmethod
    def compute_battery_state_change(battery_state):
        bsd_lag_1 = battery_state.groupby('solbox_id').shift(1)
        bsd_lag_1[bsd_lag_1.isnull().values] = battery_state[bsd_lag_1.isnull().values]
        return ((battery_state - bsd_lag_1) != 0).rename('battery_state_change')

    def add_battery_state_mappings(self, data):
        data['battery_state_discrete_at_change'] = data.apply(
            lambda x: np.nan if not x['battery_state_change'] else x['battery_state_discrete'], axis=1)
        data['battery_state_percent_at_change'] = data['battery_state_discrete_at_change'].map(
            self.discrete_to_percent)
        data['battery_state_continuous_at_change'] = data.apply(
            lambda x: x['battery_state_percent_at_change'] * self.battery_capacities[x.name[0]] / 100, axis=1)
        return data

    def drop_data_before_battery_change(self, data):
        before_change = data['battery_state_discrete_at_change'].groupby('solbox_id')\
            .fillna(method='ffill').isnull()
        return data.loc[~before_change]

    def compute_battery_state_theoretical(self, data):
        return data.groupby('solbox_id').apply(self._compute_battery_state_theoretical)['battery_state_continuous_theoretical']

    def _compute_battery_state_theoretical(self, data):
        valid_indices = self.get_start_end_tuples(data['battery_state_continuous_at_change'].dropna())
        data['battery_state_continuous_theoretical'] = data['battery_state_continuous_at_change']
        for i_0, i_1 in valid_indices:
            time_m = (data.index.get_level_values(1) > i_0[1]) & (data.index.get_level_values(1) <= i_1[1])
            battery_state_t0 = data.loc[i_0, 'battery_state_continuous_at_change']
            energy_to_battery = data.loc[time_m, 'energy_to_battery_i']
            data.loc[time_m, 'battery_state_continuous_theoretical'] = energy_to_battery.cumsum() + battery_state_t0
        return data.reset_index()[['datetime', 'battery_state_continuous_theoretical']].set_index('datetime')

    def add_energy_to_battery_untracked(self, data):
        return data.groupby('solbox_id').apply(self._add_energy_to_battery_untracked)

    def _add_energy_to_battery_untracked(self, data):
        data['energy_to_battery_untracked_i'] = 0
        data['battery_state_continuous_computed'] = data['battery_state_continuous_at_change']
        valid_indices = self.get_start_end_tuples(data['battery_state_continuous_at_change'].dropna())
        for i_0, i_1 in valid_indices:
            time_m = (data.index.get_level_values(1) > i_0[1]) & (data.index.get_level_values(1) <= i_1[1])
            battery_state_t0 = data.loc[i_0, 'battery_state_continuous_at_change']
            battery_state_t1 = data.loc[i_1, 'battery_state_continuous_at_change']
            energy_to_battery = data.loc[time_m, 'energy_to_battery_i']
            untracked_consumption = battery_state_t0 + np.sum(energy_to_battery) - battery_state_t1
            untracked_consumption_per_step = untracked_consumption / len(energy_to_battery)
            data.loc[time_m, 'energy_to_battery_untracked_i'] = -untracked_consumption_per_step
            data.loc[time_m, 'battery_state_continuous_computed'] = battery_state_t0 +\
                data.loc[time_m, 'energy_to_battery_untracked_i'].cumsum() +\
                data.loc[time_m, 'energy_to_battery_i'].cumsum()
                
        return data

    @staticmethod
    def compute_energy_production(data):
        return data['energy_production_i'] + data['energy_excess_i'] +\
            (data['energy_to_battery_untracked_i'] > 0 * data['energy_to_battery_untracked_i'])

    @staticmethod
    def compute_energy_consumption(data):
        return data['energy_consumption_i'] +\
            (data['energy_to_battery_untracked_i'] < 0 * data['energy_to_battery_untracked_i'])

    @staticmethod
    def rename_columns(data):
        data = data.rename(columns={
            'TIME_DHAKA': 'timestamp_dhaka',
            'fILoadDirect_avg': 'energy_consumption_i',
            'fIPV_avg': 'energy_production_i',
            'fIExcess_avg': 'energy_excess_i',
            'fIToBat_avg': 'energy_to_battery_i',
            'fIToGrid_avg': 'energy_to_grid_i',
            'fTemperature_avg': 'temperature',
            'fSellPrice': 'price_sell',
            'fBuyPrice': 'price_buy',
            'u8StateOfBattery': 'battery_state_discrete',
            'u32TotalBought': 'energy_bought',
            'u32TotalSold': 'energy_sold',
        })
        return data

    @staticmethod
    def add_date_features(data):
        data['datetime'] = pd.to_datetime(data['timestamp_dhaka'], format='%d/%m/%y %H:%M:%S')
        data['date'] = data['datetime'].dt.date
        data['year'] = data['datetime'].dt.year
        data['month'] = data['datetime'].dt.month
        data['week'] = data['datetime'].dt.week
        data['weekday'] = data['datetime'].dt.weekday
        data['day'] = data['datetime'].dt.day
        data['hour'] = data['datetime'].dt.hour
        return data

    @staticmethod
    def compute_w_per_hour_for_two_minutes_data(volt_var, current_var):
        return volt_var * current_var * 2 / 60

    @staticmethod
    def strings_to_floats(data):
        assert all([t in [object, str] for t in data.dtypes])
        data = data.copy()
        for v in data:
            data[v] = data[v].astype(str).apply(
                lambda x: x.replace(',', '.')).astype(float)

        return data

    def remove_outliers(self, data, percentiles=[5, 95]):
        data = data.copy()
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
    def group_data_hourly(data, stats):
        data_hourly = data.groupby(
            ['solbox_id', 'year', 'month', 'day', 'hour']).agg(stats).reset_index()
        data_hourly['datetime'] = data_hourly.apply(lambda x: datetime.datetime(
            int(x['year']), int(x['month']), int(x['day']), int(x['hour'])), axis=1)
        return data_hourly.set_index(['solbox_id', 'datetime'])
    
    @staticmethod
    def get_start_end_tuples(x):
        return ((x.index[i], x.index[i + 1]) for i in range(len(x[:-1].index)))

    @property 
    def schema(self):
        return object

