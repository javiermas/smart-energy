import json
from smartenergy.database import Stations


stations = Stations()
with open('data/grid.json') as f:
    grid = json.load(f)
    for station in grid:
        stations.update_single_station(station, station['solbox_id'])
