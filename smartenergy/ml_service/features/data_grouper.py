from datetime import datetime
import numpy as np
from pandas import date_range, DataFrame, concat

from .base import Feature


class DataGrouper(Feature):

    def __init__(self):
        super().__init__()
        sum_cols = ['energy_generation_i', 'energy_consumption_i', 'energy_excess_i',
                    'energy_to_grid_i', 'energy_to_battery_i']
        sum_stats = {key: np.sum for key in sum_cols}
        mean_stats = {key: self.scaled_sum for key in ['temperature']}
        last_stats = {key: 'last' for key in ['battery_state_discrete']}
        self.stats = {**sum_stats, **mean_stats, **last_stats}

    def transform(self, data):
        data_grouped = self.group_data_hourly(data['MinuteMeasurements'])
        data_grouped = self.fill_time_gaps(data_grouped)
        return data_grouped

    def group_data_hourly(self, data):
        data_hourly = data.groupby(
            ['solbox_id', 'year', 'month', 'day', 'hour']).agg(self.stats).reset_index()
        data_hourly['datetime'] = data_hourly.apply(lambda x: datetime(
            int(x['year']), int(x['month']), int(x['day']), int(x['hour'])), axis=1)
        return data_hourly.set_index(['solbox_id', 'datetime'])

    def fill_time_gaps(self, data):
        dates_list = list()
        for _id in data.index.get_level_values(0).unique():
            min_date, max_date = min(data.loc[_id].reset_index()['datetime']), max(data.loc[_id].reset_index()['datetime'])
            _dates = date_range(min_date, max_date, freq='H')
            dates = DataFrame({
                'datetime': _dates,
                'solbox_id': [_id] * len(_dates),
            })
            dates_list.append(dates)

        dates = concat(dates_list).set_index(['solbox_id', 'datetime'])
        return data.join(dates, how='outer')

    @staticmethod
    def scaled_sum(data):
        return np.mean(data) * 30  # 2 minutes to hours

    @property
    def schema_input(self):
        return object
