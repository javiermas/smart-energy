import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from ....ml_service.features import BatteryStateFeatures
from ...custom_assert_functions import assertFrameEqual, assertSeriesEqual


class TestBatteryStateFeatures(unittest.TestCase):

    def setUp(self):
        self.battery_state_features = BatteryStateFeatures()

    def test_compute_battery_state_change_single_station_returns_expected_output(self):
        data = pd.DataFrame({
            'battery_state_discrete': [2] * 2 + [3] * 2 + [2] * 2,
            'solbox_id': ['a'] * 6,
        }).set_index('solbox_id')
        expected_output = pd.DataFrame({
            'battery_state_change': [False, False, True, False, True, False],
            'solbox_id': ['a'] * 6,
        }).set_index('solbox_id')
        output = self.battery_state_features.compute_battery_state_change(data['battery_state_discrete'])
        assertSeriesEqual(output, expected_output['battery_state_change'])

    def test_compute_battery_state_change_multiple_stations_returns_expected_output(self):
        data = pd.DataFrame({
            'battery_state_discrete': [2] * 3 + [3] * 2 + [2] * 1,
            'solbox_id': ['a'] * 3 + ['b'] * 3,
        }).set_index('solbox_id')
        expected_output = pd.DataFrame({
            'battery_state_change': [False, False, False, False, False, True],
            'solbox_id': ['a'] * 3 + ['b'] * 3,
        }).set_index('solbox_id')
        output = self.battery_state_features.compute_battery_state_change(data['battery_state_discrete'])
        assertSeriesEqual(output, expected_output['battery_state_change'])

    def test_drop_data_before_battery_change_single_station(self):
        data = pd.DataFrame({
            'battery_state_discrete_at_change': [np.nan] * 3 + [3] * 2,
            'solbox_id': ['a'] * 5,
        }).set_index('solbox_id')
        expected_output = pd.DataFrame({
            'battery_state_discrete_at_change': [3., 3.],
            'solbox_id': ['a', 'a'],
        }).set_index('solbox_id')
        output = self.battery_state_features.drop_data_before_battery_change(data)
        assertFrameEqual(output, expected_output)

    def test_drop_data_before_battery_change_multiple_stations(self):
        data = pd.DataFrame({
            'battery_state_discrete_at_change': [np.nan] * 3 + [3] * 2 + [np.nan] * 2 + [2],
            'solbox_id': ['a'] * 5 + ['b'] * 3,
        }).set_index('solbox_id')
        expected_output = pd.DataFrame({
            'battery_state_discrete_at_change': [3., 3., 2.],
            'solbox_id': ['a', 'a', 'b'],
        }).set_index('solbox_id')
        output = self.battery_state_features.drop_data_before_battery_change(data)
        assertFrameEqual(output, expected_output)

    def test_add_energy_to_battery_untracked_single_station(self):
        stations = 1
        data, expected_output = [], []
        for i in range(stations):
            data.append(pd.DataFrame({
                'battery_state_continuous_at_change': [10] + [np.nan] * 2 + [7],
                'energy_to_battery_i': [1] * 3 + [-2],
                'solbox_id': [str(i)] * 4,
                'datetime': [datetime(2018, 1, 1, 0, 0, i * 2) for i in range(4)],
            }).set_index(['solbox_id', 'datetime']))
            expected_output.append(pd.DataFrame({
                'battery_state_continuous_at_change': [10] + [np.nan] * 2 + [7],
                'energy_to_battery_i': [1] * 3 + [-2],
                'battery_state_continuous_theoretical': [10., 11., 12., 10.],
                'solbox_id': [str(i)] * 4,
                'datetime': [datetime(2018, 1, 1, 0, 0, i * 2) for i in range(4)],
                'energy_to_battery_untracked_i': [0., -1., -1., -1.],
                'battery_state_continuous_computed': [10., 10., 10., 7.],
            }).set_index(['solbox_id', 'datetime']))

        data = pd.concat(data)
        expected_output = pd.concat(expected_output)
        output = self.battery_state_features.add_energy_to_battery_untracked(data)
        assertFrameEqual(output, expected_output)

    def test_add_energy_to_battery_untracked_multiple_stations(self):
        stations = 2
        data, expected_output = [], []
        for i in range(stations):
            data.append(pd.DataFrame({
                'battery_state_continuous_at_change': [10] + [np.nan] * 2 + [7],
                'energy_to_battery_i': [1] * 3 + [-2],
                'solbox_id': [str(i)] * 4,
                'datetime': [datetime(2018, 1, 1, 0, 0, i * 2) for i in range(4)],
            }).set_index(['solbox_id', 'datetime']))
            expected_output.append(pd.DataFrame({
                'battery_state_continuous_at_change': [10] + [np.nan] * 2 + [7],
                'energy_to_battery_i': [1] * 3 + [-2],
                'battery_state_continuous_theoretical': [10., 11., 12., 10.],
                'solbox_id': [str(i)] * 4,
                'datetime': [datetime(2018, 1, 1, 0, 0, i * 2) for i in range(4)],
                'energy_to_battery_untracked_i': [0., -1., -1., -1.],
                'battery_state_continuous_computed': [10., 10., 10., 7.],
            }).set_index(['solbox_id', 'datetime']))

        data = pd.concat(data)
        expected_output = pd.concat(expected_output)
        output = self.battery_state_features.add_energy_to_battery_untracked(data)
        assertFrameEqual(output, expected_output)
