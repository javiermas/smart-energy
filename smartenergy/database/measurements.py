from pandas import DataFrame, concat
from .mongo_collection import MongoCollection


class Measurements(MongoCollection):

    def __init__(self, collection, connection):
        super(Measurements, self).__init__(collection, connection)

    def load_single_station(self, station_id, nrows=None):
        return DataFrame(list(self.coll.find({'solbox_id': station_id}).limit(nrows)))

    def load_all_stations_first_n(self, nrows=None):
        data = list()
        for station in self.station_ids:
            data.append(self.load_single_station(self.coll, station, nrows))

        return concat(data)

    def get_generator_measurement(self, t):
        return 1

    def get_battery_measurement(self, t):
        return 1

    def get_consumer_measurement(self, t):
        return 1

    def get_grid_measurement(self, t, station_id):
        return 1

    @property
    def station_ids(self):
        return list(self.coll.distinct('solbox_id'))
