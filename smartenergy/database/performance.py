from .mongo_collection import MongoCollection
from .connection import Connection


class Performance(MongoCollection):

    def __init__(self, connection=Connection.get_connection()):
        super().__init__('performance', connection)
