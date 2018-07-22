from pandas import DataFrame
from .mongo_collection import MongoCollection
from .connection import Connection


class Stations(MongoCollection):

    def __init__(self, connection=Connection()):
        super().__init__('stations', connection)

    def load_single_station(self, station_id):
        return DataFrame(list(self.coll.find({'solbox_id': station_id}))).set_index('solbox_id')

    @property
    def station_ids(self):
        return list(self.coll.distinct('solbox_id'))
