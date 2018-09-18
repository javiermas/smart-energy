from pandas import DataFrame, concat

from .mongo_collection import MongoCollection


class Measurements(MongoCollection):

    def __init__(self, collection, connection):
        super().__init__(collection, connection)

    def load(self, nrows=0):
        return DataFrame(list(self.coll.find().limit(nrows))).drop('_id', axis=1)

    def load_single_station(self, station_id, limit=0):
        return DataFrame(list(self.coll.find({'solbox_id': station_id}).limit(limit)))

    def load_multiple_stations(self, station_ids, limit=0):
        data = [self.load_single_station(station, limit) for station in station_ids]
        return concat(data)

    def load_all_stations(self, limit=0):
        data = [self.load_single_station(station, limit) for station in self.station_ids]
        return concat(data)

    def load_data_within(self, start, end):
        return DataFrame(list(self.coll.find({'datetime': {'$gte': start, '$lt': end}}))).drop('_id', axis=1)

    def load_until(self, end=None, nrows=0):
        return DataFrame(list(self.coll.find({'datetime': {'$lt': end}}).limit(nrows))).drop('_id', axis=1)

    def load_from(self, start=None, nrows=0):
        return DataFrame(list(self.coll.find({'datetime': {'$gt': start}}).limit(nrows))).drop('_id', axis=1)

    def load_first_measurement(self):
        return list(self.coll.find().sort('datetime', 1).limit(1))

    def get_last_measurement_single_station(self, station_id):
        return list(self.coll.find({'solbox_id': station_id}).sort('datetime', -1).limit(1))

    def get_last_field_single_station(self, station_id, field):
        m = list(self.coll.find({'solbox_id': station_id}, {field: 1}).sort('datetime', -1).limit(1))
        if not m:
            return None

        return m[0][field]

    def get_last_generator_measurement(self, station_id):
        return self.get_last_field_single_station(station_id, 'energy_generation_computed_i')

    def get_last_consumer_measurement(self, station_id):
        return self.get_last_field_single_station(station_id, 'energy_consumption_computed_i')

    def get_last_battery_measurement(self, station_id):
        return self.get_last_field_single_station(station_id, 'battery_state_percent')

    def get_last_excess_energy_measurement(self, station_id):
        return self.get_last_field_single_station(station_id, 'energy_excess_i')

    @property
    def station_ids(self):
        return list(self.coll.distinct('solbox_id'))
