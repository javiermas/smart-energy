import unittest
import pandas as pd
import numpy as np

from ....ml_service.features import BasicPreprocessor
from ...custom_assert_functions import assertFrameEqual, assertSeriesEqual


class TestBasicPreprocessor(unittest.TestCase):

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
