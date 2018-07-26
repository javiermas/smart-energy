from pandas import DataFrame
from abc import abstractmethod
from .mongo_collection import MongoCollection


class Predictions(MongoCollection):

    def __init__(self, collection, connection):
        super().__init__(collection, connection)

    def load(self, nrows=0):
        return DataFrame(list(self.coll.find().limit(nrows))).drop('_id', axis=1)

    @abstractmethod
    def load_predictions(self):
        pass

    def load_single_station(self, station_id, nrows=0):
        return DataFrame(list(self.coll.find({'solbox_id': station_id}).limit(nrows)))

    def store(self):
        pass

    @property
    def station_ids(self):
        return list(self.coll.distinct('solbox_id'))
