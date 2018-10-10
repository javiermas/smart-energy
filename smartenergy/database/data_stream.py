from pandas import DataFrame
from numpy import nan


class DataStream(DataFrame):

    t = None
    last_measurement = {}
    measurements = ['energy_generation_computed_i', 'energy_consumption_computed_i',
                    'battery_state_percent', 'energy_excess_i']
    
    @classmethod
    def initialize(cls, data):
        stream = cls.from_dict(data)
        stream.set_index(['solbox_id', 'datetime'], inplace=True)
        return stream

    def refresh(self, t):
        self.t = t
        for installation in self.index.get_level_values(0).unique():
            try:
                _last_measurement = self.loc[(installation, self.t), self.measurements]\
                    .astype(float).round(2).to_dict()
                self.last_measurement[installation] = _last_measurement
            except KeyError:
                _last_measurement = {measurement: nan for measurement in self.measurements}
                self.last_measurement[installation] = _last_measurement

    def get_last_generator_measurement(self, installation):
        return self.last_measurement[installation]['energy_generation_computed_i']

    def get_last_consumer_measurement(self, installation):
        return self.last_measurement[installation]['energy_consumption_computed_i']

    def get_last_battery_measurement(self, installation):
        return self.last_measurement[installation]['battery_state_percent']

    def get_last_energy_excess_measurement(self, installation):
        return self.last_measurement[installation]['energy_excess_i']

    def get_all_measurements_since(self, datetime):
        return self.loc[(slice(None), datetime), :]

    def get_all_measurements(self):
        return self.reset_index()

    def get_first_datetime(self):
        return self.index.get_level_values('datetime').min()

    def get_last_datetime(self):
        return self.index.get_level_values('datetime').max()
