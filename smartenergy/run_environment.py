from smartenergy.network import Network, Pipe, installation
from smartenergy.ml_service import MLService
from smartenergy.environments import SBEnvironment
from smartenergy.database import HourlyMeasurements, Stations


init_t = 0
step_size = 1

stations = Stations()
station_ids = stations.station_ids
network = Network()
for _id in station_ids:
    row = stations.load_single_station(_id)
    installation_elements = [
        installation.Generator({
            'res_gen_bat': Pipe(row['res_gen_bat']),
            'res_gen_con': Pipe(row['res_gen_con']),
            'res_gen_grid': Pipe(row['res_gen_grid']),
        }),
        installation.Battery({
            'res_bat_con': Pipe(row['res_bat_con']),
            'res_bat_grid': Pipe(row['res_bat_grid']),
        }),
        installation.Consumer(),
    ]
    connection_to_grid = {k: Pipe(v) for k, v in row.items() if 'res_self_' in k}
    _installation = installation.Installation(_id, installation_elements, connection_to_grid)
    network.add_installation(_installation)

ml_service = MLService(network.action_space)

sb_environment = SBEnvironment(
    ml_service=ml_service,
    network=network,
    init_t=init_t,
    step_size=step_size
)

sb_environment.run(steps=100)


