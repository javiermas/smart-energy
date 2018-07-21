from pymongo import MongoClient


class Connection(object):

    def __init__(self):
        pass
    
    @staticmethod
    def get_connection(host='127.0.0.1', port=27017):
        client = MongoClient(host=host, port=port)
        db = client['energy']
        return db
