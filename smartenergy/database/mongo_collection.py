from abc import ABC


class MongoCollection(ABC):

    def __init__(self, coll, connection):
        self.coll = connection[coll]
        self.connection = connection

    def load(self):
        return list(self.coll.find())

    def insert_one(self, data):
        self.coll.insert_one(data)

    def insert_many(self, data):
        self.coll.insert_many(data)

    def drop(self):
        self.coll.drop()
