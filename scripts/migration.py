import logging
import pandas as pd
from smartenergy.database import MinuteMeasurements, HourlyMeasurements
from smartenergy.ml_service import features


# Load minute-level data
logging.info('Loading minute-level data')
minute_measurements = MinuteMeasurements()
stations = minute_measurements.station_ids
data = {'MinuteMeasurements': minute_measurements.load_multiple_stations(stations)}
logging.info(f'{data["MinuteMeasurements"].shape[0]} minute-level records found')

# Group data
migration_pipeline = features.Pipeline(
    preprocessors=[
        features.BasicPreprocessor(),
        features.DateFeatures(),
    ],
    features=[
        features.DataGrouper(),
        features.BatteryStateFeatures(),
    ]
)

data = migration_pipeline.transform(data)
data_hourly = data['DataGrouper']
data_hourly = pd.merge(data['DataGrouper'], data['BatteryStateFeatures'],
                       on=['solbox_id', 'datetime'], how='outer')

# Store hour-level data
logging.info(f'{data_hourly.shape[0]} hour-level records will be stored')
hourly_measurements = HourlyMeasurements()
hourly_measurements.drop()
hourly_measurements.insert_many(data_hourly.reset_index().to_dict('records'))
