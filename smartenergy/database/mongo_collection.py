from abc import ABC
from pandas import DataFrame


class MongoCollection(ABC):

    def __init__(self, coll, connection):
        self.coll = connection[coll]
        self.connection = connection

    def load(self, nrows=0):
        return DataFrame(list(self.coll.find().limit(nrows))).drop('_id', axis=1)

    def insert_many(self, data):
        self.coll.insert_many(data)

    def drop(self):
        self.coll.drop()
