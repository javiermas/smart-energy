import time
import pandas as pd


def get_station_ids(coll):
    return list(coll.distinct('solbox_id'))


def load_single_station(coll, station_id, nrows=None):
    return pd.DataFrame(list(coll.find({'solbox_id': station_id}).limit(nrows)))


def load_all_stations(coll, nrows=None):
    return pd.DataFrame(list(coll.find().limit(nrows)))


def load_all_stations_first_n(coll, nrows=None):
    data = list()
    for station in get_station_ids(coll):
        data.append(load_single_station(coll, station, nrows))

    return pd.concat(data)


def insert_many(data, coll):
    coll.insert_many(data.to_dict('records'))


def store_hourly_predictions(coll, model, solbox_id, datetime, prediction, ground_truth):
    coll.insert({
        'model': model,
        'solbox_id': solbox_id,
        'year': datetime.year,
        'month': datetime.month,
        'day': datetime.day,
        'hour': datetime.hour,
        'prediction': prediction,
        'ground_truth': ground_truth,
        't_stored': time.time(),
    })


def drop_predictions(coll):
    coll.drop()


def load_predictions(coll):
    return pd.DataFrame(list(coll.find())).set_index(['solbox_id', 'year', 'month', 'day', 'hour']).sort_index()
