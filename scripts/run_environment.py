from argparse import ArgumentParser
from datetime import timedelta

from smartenergy.network import Network, installation
from smartenergy.ml_service import MLService, FeatureService, ForecastService, AgentService
from smartenergy.ml_service.features import LaggedReadings
from smartenergy.ml_service.models import XGBoostHourlyGenerationStationPredictor, NeuralAgent
from smartenergy.environments.sb_environment import SBEnvironment
from smartenergy.database import HourlyMeasurements, Stations, DataStream

parser = ArgumentParser()
parser.add_argument('--burning_steps')
parser.add_argument('--init_steps')
parser.add_argument('--simulation_steps')
args = parser.parse_args()

hourly_measurements = HourlyMeasurements()
data_stream = DataStream.initialize(hourly_measurements.load_all())
stations = Stations()
station_ids = stations.station_ids
network = Network()
for _id in station_ids:
    if _id == '311':
        continue

    station = stations.load_single_station(_id)
    installation_elements = {
        'generator': installation.Generator(data_stream),
        'battery': installation.Battery(data_stream, station['battery_capacity']*60/2/15*1000),
        'consumer': installation.Consumer(data_stream),
    }
    _installation = installation.Installation(_id, installation_elements, station['connections'])
    network.add_installation(_installation)

#  ML_service
if args.burning_steps is None:
    burning_steps = 24 * 7
else:
    burning_steps = eval(args.burning_steps)

if args.init_steps is None:
    init_steps = 24 * 7
else:
    init_steps = eval(args.init_steps)

if args.simulation_steps is None:
    simulation_steps = 24 * 7
else:
    simulation_steps = eval(args.simulation_steps)

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
    data_stream=data_stream,
)

sb_environment = SBEnvironment(
    ml_service=ml_service,
    network=network,
    burning_steps=burning_steps,
    init_steps=init_steps,
    step_size=step_size,
    data_stream=data_stream,
)

sb_environment.run(steps=simulation_steps)
