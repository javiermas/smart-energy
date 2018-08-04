from pandas import DataFrame, concat
from .mongo_collection import MongoCollection


class Measurements(MongoCollection):

    def __init__(self, collection, connection):
        super(Measurements, self).__init__(collection, connection)

    def load(self, nrows=0):
        return DataFrame(list(self.coll.find().limit(nrows))).drop('_id', axis=1)

    def load_single_station(self, station_id, nrows=None):
        return DataFrame(list(self.coll.find({'solbox_id': station_id}).limit(nrows)))

    def load_all_stations_first_n(self, nrows=None):
        data = list()
        for station in self.station_ids:
            data.append(self.load_single_station(self.coll, station, nrows))

        return concat(data)

    def load_until(self, end=None, nrows=0):
        return DataFrame(list(self.coll.find({'datetime': {'$lt': end}}).limit(nrows))).drop('_id', axis=1)

    def load_from(self, start=None, nrows=0):
        return DataFrame(list(self.coll.find({'datetime': {'$gt': start}}).limit(nrows))).drop('_id', axis=1)

    def get_first_measurement(self):
        return list(self.coll.find().sort('datetime', 1).limit(1))

    def get_last_measurement_single_station(self, station_id):
        return list(self.coll.find({'solbox_id': station_id}).sort('datetime', 1).limit(1))
    
    def get_last_field_single_station(self, station_id, field): 
        m = list(self.coll.find({'solbox_id': station_id}, {field}).sort('datetime', 1).limit(1))
        if not m:
            return None

        return m[0][field]

    def get_last_generator_measurement(self, station_id):
        return self.get_last_field_single_station(station_id, 'fILoadDirect_avg')

    def get_last_consumer_measurement(self, station_id):
        return self.get_last_field_single_station(station_id, 'fIPV_avg')

    def get_last_battery_measurement(self, station_id):
        return self.get_last_field_single_station(station_id, 'u8StateOfBattery')

    @property
    def station_ids(self):
        return list(self.coll.distinct('solbox_id'))
