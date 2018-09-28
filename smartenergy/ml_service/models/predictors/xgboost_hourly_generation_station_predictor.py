from ....database import Stations
from .xgboost_predictor import XGBoostPredictor


DEFAULT_HYPERPARAMETERS = {
    'n_jobs': 8
}


class XGBoostHourlyGenerationStationPredictor(XGBoostPredictor):

    def __init__(self, station_id, hyperparameters=None, station_ids=Stations().station_ids):
        hyperparameters = hyperparameters or DEFAULT_HYPERPARAMETERS
        self.station_id = station_id
        self.station_ids = station_ids
        super().__init__(hyperparameters)

    def __repr__(self):
        return f'XGBoost_hourly_generation_{self.station_id}'

    @property
    def feature_schema(self):
        lags = 2
        measurements = ['energy_excess_i', 'energy_generation_computed_i',
                        'energy_consumption_computed_i', 'energy_to_battery_i',
                        'energy_to_grid_i', 'temperature', 'battery_state_percent']
        lagged_measurements = {f'{m}_lag_{lag}_station{station}': float for m in measurements
                               for station in self.station_ids for lag in range(1, lags)}
        return lagged_measurements

    @property
    def target_schema(self):
        return {
            f'energy_generation_computed_i_lag_0_station{self.station_id}': float
        }
