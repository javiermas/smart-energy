from pandas import DataFrame
from .mongo_collection import MongoCollection
from .connection import Connection


class Memories(MongoCollection):

    def __init__(self, connection=Connection.get_connection()):
        super().__init__('memories', connection)

    def load_all_memories(self, limit=0):
        return list(self.coll.find().limit(limit))

    def load_n_random_memories(self, n):
        return list(self.coll.aggregate([{'$sample': {'size': n}}]))
