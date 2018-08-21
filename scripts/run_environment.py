from datetime import timedelta
from smartenergy.network import Network, Pipe, installation
from smartenergy.ml_service import MLService, FeatureService, ForecastService, AgentService
from smartenergy.ml_service.features import LaggedReadings
from smartenergy.ml_service.models import XGBoostHourlyGenerationStationPredictor, BasicAgent
from smartenergy.environments.sb_environment import SBEnvironment
from smartenergy.database import HourlyMeasurements, Stations


stations = Stations()
station_ids = stations.station_ids
network = Network()
for _id in station_ids:
    row = stations.load_single_station(_id)
    installation_elements = {
        'generator': installation.Generator({
            'res_gen_bat': Pipe(row['res_gen_bat']),
            'res_gen_con': Pipe(row['res_gen_con']),
            'res_gen_grid': Pipe(row['res_gen_grid']),
        }),
        'battery': installation.Battery({
            'res_bat_con': Pipe(row['res_bat_con']),
            'res_bat_grid': Pipe(row['res_bat_grid']),
        }),
        'consumer': installation.Consumer(),
    }
    connection_to_grid = {k: Pipe(v) for k, v in row.items() if 'res_self_' in k}
    _installation = installation.Installation(_id, installation_elements, connection_to_grid)
    network.add_installation(_installation)

#  ML_service
hourly_measurements = HourlyMeasurements()
init_t = hourly_measurements.get_first_measurement()[0]['datetime'] + timedelta(days=10)
step_size = timedelta(hours=1)

lags = 24
features = [LaggedReadings(lags=24)]
feature_service = FeatureService(features)
models = [XGBoostHourlyGenerationStationPredictor(station_id=station_id) for station_id in station_ids]
forecast_service = ForecastService(models)
action_space = {
    f'installation_{_id}': {
        'generator': range(2),  # 'generator_covered_energy'
        'battery': range(3),  # 'transaction_with_energy'
    } for _id in station_ids
}

agent_parameters = {
    'epsilon': 0.1,
}
network_parameters = {
    'hidden_units': 5,
    'learning_rate': 0.01,
}

non_tunable_parameters = {
    'action_space': action_space,
}

parameters = {**agent_parameters, **network_parameters, **non_tunable_parameters}

basic_agent = BasicAgent(**parameters)
agent_service = AgentService(basic_agent)

ml_service = MLService(
    feature_service=feature_service,
    forecast_service=forecast_service,
    agent_service=agent_service,
)

sb_environment = SBEnvironment(
    ml_service=ml_service,
    network=network,
    init_t=init_t,
    step_size=step_size
)

sb_environment.run(steps=100)
