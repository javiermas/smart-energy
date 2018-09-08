import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from ...database import BasicPreprocessor
from ..custom_assert_functions import assertFrameEqual, assertSeriesEqual


class TestPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = BasicPreprocessor()

    def test_strings_to_floats_with_int_raises_exception(self):
        data = pd.DataFrame({
            'var_0': [0.2, 3.3, 3333333.3],
            'var_1': [0.2, 3.3, 3333333.3],
        })
        self.assertRaises(AssertionError, self.preprocessor.strings_to_floats, data=data)

    def test_strings_to_floats_with_dataframe_returns_expected_output(self):
        data = pd.DataFrame({
            'var_0': ['0,2', '3,3', '3333333,3'],
            'var_1': ['0,2', '3,3', '3333333,3'],
        })
        expected_output = pd.DataFrame({
            'var_0': [0.2, 3.3, 3333333.3],
            'var_1': [0.2, 3.3, 3333333.3],
        })
        output = self.preprocessor.strings_to_floats(data)
        assertFrameEqual(output, expected_output)

    def test_remove_outliers_single_station_returns_expected_output(self):
        data = pd.DataFrame({
            'var_0': [0.2] * 1000 + [3.3] * 1000 + [3333333.3] * 10,
            'var_1': [0.2] * 1000 + [3.3] * 1000 + [3333333.3] * 10,
            'solbox_id': ['a'] * 2010,
        }).set_index('solbox_id')
        expected_output = pd.DataFrame({
            'var_0': [0.2] * 1000 + [3.3] * 1000 + [np.nan] * 10,
            'var_1': [0.2] * 1000 + [3.3] * 1000 + [np.nan] * 10,
            'solbox_id': ['a'] * 2010,
        }).set_index('solbox_id')
        output = self.preprocessor.remove_outliers(data)
        assertFrameEqual(output, expected_output)

    def test_remove_outliers_multiple_stations_returns_expected_output(self):
        data = pd.DataFrame({
            'var_0': [0.2] * 1000 + [3.3] * 1000 + [3333333.3] * 10,
            'var_1': [0.2] * 1000 + [3.3] * 1000 + [3333333.3] * 10,
            'solbox_id': ['a'] * 1000 + ['b'] * 1010,
        }).set_index('solbox_id')
        expected_output = pd.DataFrame({
            'var_0': [0.2] * 1000 + [3.3] * 1000 + [np.nan] * 10,
            'var_1': [0.2] * 1000 + [3.3] * 1000 + [np.nan] * 10,
            'solbox_id': ['a'] * 1000 + ['b'] * 1010,
        }).set_index('solbox_id')
        output = self.preprocessor.remove_outliers(data)
        assertFrameEqual(output, expected_output)

    def test_add_offset_single_station_returns_expected_output(self):
        data = pd.DataFrame({
            'var_0': [0.2] * 1000 + [3.3] * 1000 + [-10] * 10,
            'var_1': [0.2] * 1000 + [3.3] * 1000 + [-10] * 10,
            'solbox_id': ['a'] * 2010,
        }).set_index('solbox_id')
        expected_output = pd.DataFrame({
            'var_0': [10.2] * 1000 + [13.3] * 1000 + [0] * 10,
            'var_1': [10.2] * 1000 + [13.3] * 1000 + [0] * 10,
            'solbox_id': ['a'] * 2010,
        }).set_index('solbox_id')
        output = self.preprocessor.add_offset(data)
        assertFrameEqual(output, expected_output)

    def test_add_offset_multiple_stations_returns_expected_output(self):
        data = pd.DataFrame({
            'var_0': [0.2] * 1000 + [3.3] * 1000 + [-10] * 10,
            'var_1': [0.2] * 1000 + [3.3] * 1000 + [-10] * 10,
            'solbox_id': ['a'] * 1000 + ['b'] * 1010,
        }).set_index('solbox_id')
        expected_output = pd.DataFrame({
            'var_0': [0.2] * 1000 + [13.3] * 1000 + [0] * 10,
            'var_1': [0.2] * 1000 + [13.3] * 1000 + [0] * 10,
            'solbox_id': ['a'] * 1000 + ['b'] * 1010,
        }).set_index('solbox_id')
        output = self.preprocessor.add_offset(data)
        assertFrameEqual(output, expected_output)

    def test_compute_battery_state_change_single_station_returns_expected_output(self):
        data = pd.DataFrame({
            'battery_state_discrete': [2] * 2 + [3] * 2 + [2] * 2,
            'solbox_id': ['a'] * 6,
        }).set_index('solbox_id')
        expected_output = pd.DataFrame({
            'battery_state_change': [False, False, True, False, True, False],
            'solbox_id': ['a'] * 6,
        }).set_index('solbox_id')
        output = self.preprocessor.compute_battery_state_change(data['battery_state_discrete'])
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
        output = self.preprocessor.compute_battery_state_change(data['battery_state_discrete'])
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
        output = self.preprocessor.drop_data_before_battery_change(data)
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
        output = self.preprocessor.drop_data_before_battery_change(data)
        assertFrameEqual(output, expected_output)

    def test_compute_battery_state_theoretical_single_station(self):
        data = pd.DataFrame({
            'battery_state_continuous_at_change': [10] + [np.nan] * 2 + [7],
            'energy_to_battery_i': [1] * 3 + [-2],
            'solbox_id': ['a'] * 4,
            'datetime': [datetime(2018, 1, 1, 0, 0, i * 2) for i in range(4)],
        }).set_index(['solbox_id', 'datetime'])
        expected_output = pd.DataFrame({
            'battery_state_continuous_theoretical': [10., 11., 12., 10.],
            'solbox_id': ['a'] * 4,
        }).set_index('solbox_id')
        output = self.preprocessor.compute_battery_state_theoretical(data)
        assertSeriesEqual(output, expected_output['battery_state_continuous_theoretical'])

    def test_battery_state_theoretical_multiple_stations(self):
        users = 2
        data, expected_output = [], []
        for i in range(users):
            data.append(pd.DataFrame({
                'battery_state_continuous_at_change': [10] + [np.nan] * 2 + [7],
                'energy_to_battery_i': [1] * 3 + [-2],
                'solbox_id': [str(i)] * 4,
                'datetime': [datetime(2018, 1, 1, 0, 0, i * 2) for i in range(4)],
            }).set_index(['solbox_id', 'datetime']))
            expected_output.append(pd.DataFrame({
                'battery_state_continuous_theoretical': [10., 11., 12., 10.],
                'solbox_id': [str(i)] * 4,
            }).set_index('solbox_id'))

        data = pd.concat(data)
        expected_output = pd.concat(expected_output)
        output = self.preprocessor.compute_battery_state_theoretical(data)
        assertSeriesEqual(output, expected_output['battery_state_continuous_theoretical'])

    def test_add_energy_to_battery_untracked_single_station(self):
        stations = 1
        data, expected_output = [], []
        for i in range(stations):
            data.append(pd.DataFrame({
                'battery_state_continuous_at_change': [10] + [np.nan] * 2 + [7],
                'energy_to_battery_i': [1] * 3 + [-2],
                'battery_state_continuous_theoretical': [10., 11., 12., 10.],
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
        output = self.preprocessor.add_energy_to_battery_untracked(data)
        assertFrameEqual(output, expected_output)

    def test_add_energy_to_battery_untracked_multiple_stations(self):
        stations = 2
        data, expected_output = [], []
        for i in range(stations):
            data.append(pd.DataFrame({
                'battery_state_continuous_at_change': [10] + [np.nan] * 2 + [7],
                'energy_to_battery_i': [1] * 3 + [-2],
                'battery_state_continuous_theoretical': [10., 11., 12., 10.],
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
        output = self.preprocessor.add_energy_to_battery_untracked(data)
        assertFrameEqual(output, expected_output)
