import time
from pandas import DataFrame
from .predictions import Predictions
from .connection import Connection


class HourlyPredictions(Predictions):

    def __init__(self, connection=Connection().get_connection()):
        super().__init__('hourly_predictions', connection)

    def load_predictions(self):
        predictions = DataFrame(list(self.coll.find()))
        return predictions.set_index(['solbox_id', 'datetime']).sort_index()

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
