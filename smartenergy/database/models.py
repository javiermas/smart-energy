from .mongo_collection import MongoCollection
from .connection import Connection


class Models(MongoCollection):

    def __init__(self, connection=Connection.get_connection()):
        super().__init__('models', connection)

    def load(self):
        pass
