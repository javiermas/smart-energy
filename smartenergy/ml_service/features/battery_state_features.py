import numpy as np
from .base import Feature


class BatteryStateFeatures(Feature):

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
    battery_cols = ['battery_state_continuous_at_change', 'battery_state_continuous_theoretical',
                    'battery_state_continuous_computed', 'energy_to_battery_untracked_i',
                    'energy_generation_computed_i', 'energy_consumption_computed_i',
                    'battery_state_percent']

    def transform(self, data):
        data = data['DataGrouper'].copy().sort_index()
        data['battery_state_change'] = self.compute_battery_state_change(data['battery_state_discrete'])
        data = self.add_battery_state_mappings(data)
        data = self.add_energy_to_battery_untracked(data)
        data['energy_generation_computed_i'] = self.compute_energy_generation(data)
        data['energy_consumption_computed_i'] = self.compute_energy_consumption(data)
        return data[self.battery_cols].sort_index()

    @staticmethod
    def compute_battery_state_change(battery_state):
        bsd_lag_1 = battery_state.groupby('solbox_id').shift(1)
        bsd_lag_1[bsd_lag_1.isnull().values] = battery_state[bsd_lag_1.isnull().values]
        return ((battery_state - bsd_lag_1) != 0).rename('battery_state_change')

    def add_battery_state_mappings(self, data):
        data['battery_state_percent'] = data['battery_state_discrete'].map(self.discrete_to_percent)
        data['battery_state_discrete_at_change'] = data.apply(
            lambda x: np.nan if not x['battery_state_change'] else x['battery_state_discrete'], axis=1)
        data['battery_state_percent_at_change'] = data['battery_state_discrete_at_change'].map(
            self.discrete_to_percent)
        data['battery_state_continuous_at_change'] = data.apply(
            lambda x: x['battery_state_percent_at_change'] * self.battery_capacities[x.name[0]] / 100, axis=1)
        return data

    def add_energy_to_battery_untracked(self, data):
        return data.groupby('solbox_id').apply(self._add_energy_to_battery_untracked)

    def _add_energy_to_battery_untracked(self, data):
        data['energy_to_battery_untracked_i'] = 0
        data['battery_state_continuous_computed'] = data['battery_state_continuous_at_change']
        data['battery_state_continuous_theoretical'] = data['battery_state_continuous_at_change']
        valid_indices = self.get_start_end_tuples(data['battery_state_continuous_at_change'].dropna())
        for i_0, i_1 in valid_indices:
            time_m = (data.index.get_level_values(1) > i_0[1]) & (data.index.get_level_values(1) <= i_1[1])
            battery_state_t0 = data.loc[i_0, 'battery_state_continuous_at_change']
            battery_state_t1 = data.loc[i_1, 'battery_state_continuous_at_change']
            energy_to_battery = data.loc[time_m, 'energy_to_battery_i']
            untracked_consumption = battery_state_t0 + np.sum(energy_to_battery) - battery_state_t1
            untracked_consumption_per_step = untracked_consumption / len(energy_to_battery)
            data.loc[time_m, 'energy_to_battery_untracked_i'] = -untracked_consumption_per_step
            data.loc[time_m, 'battery_state_continuous_theoretical'] = energy_to_battery.cumsum() +\
                battery_state_t0
            data.loc[time_m, 'battery_state_continuous_computed'] = battery_state_t0 +\
                data.loc[time_m, 'energy_to_battery_untracked_i'].cumsum() +\
                data.loc[time_m, 'energy_to_battery_i'].cumsum()
                
        return data

    @staticmethod
    def compute_energy_generation(data):
        return data['energy_generation_i'] + data['energy_excess_i'] +\
            (data['energy_to_battery_untracked_i'] > 0 * data['energy_to_battery_untracked_i'])

    @staticmethod
    def compute_energy_consumption(data):
        return data['energy_consumption_i'] +\
            (data['energy_to_battery_untracked_i'] < 0 * data['energy_to_battery_untracked_i'])

    @staticmethod
    def get_start_end_tuples(x):
        return ((x.index[i], x.index[i + 1]) for i in range(len(x[:-1].index)))

    @staticmethod
    def drop_data_before_battery_change(data):
        before_change = data['battery_state_discrete_at_change'].groupby('solbox_id')\
            .fillna(method='ffill').isnull()
        return data.loc[~before_change]

    @property
    def schema_input(self):
        return object
