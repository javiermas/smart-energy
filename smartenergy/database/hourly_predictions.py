import time
from pandas import DataFrame
from .predictions import Predictions
from .connection import Connection


class HourlyPredictions(Predictions):

    def __init__(self, connection):
        super(HourlyPredictions, self).__init__('hourly_predictions', Connection().get_connection())

    def load_predictions(self):
        predictions = DataFrame(list(self.coll.find()))
        return predictions.set_index(['solbox_id', 'year', 'month', 'day', 'hour']).sort_index()

    def store(self, model, target, solbox_id, datetime, prediction, ground_truth):
        self.coll.insert({
            'model': model,
            'target': target,
            'solbox_id': solbox_id,
            'datetime': datetime,
            'prediction': prediction,
            'ground_truth': ground_truth,
            't_stored': time.time(),
        })
