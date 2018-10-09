from argparse import ArgumentParser
from datetime import timedelta

from smartenergy.network import Network, installation
from smartenergy.ml_service import MLService, FeatureService, ForecastService, AgentService
from smartenergy.ml_service.features import LaggedReadings
from smartenergy.ml_service.models import XGBoostHourlyGenerationStationPredictor, NeuralAgent
from smartenergy.environments.sb_environment import SBEnvironment
from smartenergy.database import HourlyMeasurements, Stations

parser = ArgumentParser()
parser.add_argument('--mode')
args = parser.parse_args()

stations = Stations()
station_ids = stations.station_ids
network = Network()
for _id in station_ids:
    if _id == '311':
        continue

    station = stations.load_single_station(_id)
    installation_elements = {
        'generator': installation.Generator({}),
        'battery': installation.Battery({}, station['battery_capacity']*60/2/15*1000),
        'consumer': installation.Consumer(),
    }
    _installation = installation.Installation(_id, installation_elements, station['connections'])
    network.add_installation(_installation)

#  ML_service
hourly_measurements = HourlyMeasurements()
if args.mode == 'test':
    burning_steps = 24
    init_steps = 24
else:
    burning_steps = 24 * 7
    init_steps = 24 * 7

step_size = timedelta(hours=1)
lags = 3
features = [LaggedReadings(lags=lags)]
feature_service = FeatureService(features)
models = [XGBoostHourlyGenerationStationPredictor(station_id=station_id) for station_id in station_ids]
forecast_service = ForecastService(models)
action_space = {
    f'installation_{_id}': {
        'generator': range(1),  # 'generator_covered_energy'
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

parameters = {**agent_parameters, **
              network_parameters, **non_tunable_parameters}

basic_agent = NeuralAgent(**parameters)
agent_service = AgentService(basic_agent)

ml_service = MLService(
    feature_service=feature_service,
    forecast_service=forecast_service,
    agent_service=agent_service,
)

sb_environment = SBEnvironment(
    ml_service=ml_service,
    network=network,
    burning_steps=burning_steps,
    init_steps=init_steps,
    step_size=step_size
)

sb_environment.run(steps=24*7)
