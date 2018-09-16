from pandas import to_datetime

from .base import Feature


class DateFeatures(Feature):

    def transform(self, data):
        return {key: self.add_date_features(value) for key, value in data.items()}

    @staticmethod
    def add_date_features(data):
        data['datetime'] = to_datetime(data['timestamp'], format='%d/%m/%y %H:%M:%S')
        data['date'] = data['datetime'].dt.date
        data['year'] = data['datetime'].dt.year
        data['month'] = data['datetime'].dt.month
        data['week'] = data['datetime'].dt.week
        data['weekday'] = data['datetime'].dt.weekday
        data['day'] = data['datetime'].dt.day
        data['hour'] = data['datetime'].dt.hour
        return data

    @property
    def schema_input(self):
        return object
