from pandas import Series
from .mongo_collection import MongoCollection
from .connection import Connection


class Stations(MongoCollection):

    def __init__(self, connection=Connection().get_connection()):
        super().__init__('stations', connection)

    def load_single_station(self, station_id):
        station_data = Series(list(self.coll.find({'solbox_id': station_id}))[0])
        return station_data.drop('_id')

    @property
    def station_ids(self):
        return list(self.coll.distinct('solbox_id'))
