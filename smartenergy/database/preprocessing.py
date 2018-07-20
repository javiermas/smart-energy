import datetime
import numpy as np
import pandas as pd


def preprocess(data):
    data['datetime'] = pd.to_datetime(data['TIME_DHAKA'], format='%d/%m/%y %H:%M:%S')
    data = add_date_features(data)
    data['solbox_id'] = data['solbox_id'].astype(str)
    data = data.set_index(['solbox_id', 'datetime']).sort_index()
    
    vars_to_float = ['fILoadDirect_avg', 'fILoad_avg', 'fIPV_avg', 'fIToGrid_avg', 'fIExcess_avg',
                     'fIToBat_avg', 'fTemperature_avg', 'fSellPrice', 'fBuyPrice']
    for v in vars_to_float:
        try:
            data[v] = data[v].apply(lambda x: x.replace(',', '.')).astype(float)
        except AttributeError:
            data[v] = data[v].apply(lambda x: x.replace(',', '.') if isinstance(x, str) else x).astype(float)

    for v in ['fILoadDirect_avg', 'fILoad_avg', 'fIPV_avg', 'fIToGrid_avg', 'fIToBat_avg']:
        data[v] = data.groupby(['solbox_id'])[v].apply(remove_outliers_lambda).ffill().fillna(0)

    for v in ['fIExcess_avg', 'fIPV_avg', 'fILoadDirect_avg', 'fILoad_avg']:
        data[v] = add_offset(data[v])

    currents = ['fILoadDirect_avg', 'fILoad_avg', 'fIPV_avg', 'fIToGrid_avg', 'fIExcess_avg', 'fIToBat_avg']
    date_cols = ['year', 'month', 'week', 'weekday', 'day', 'hour']
    other_cols = ['u8UserMode', 'u8StateOfBattery', 'u32TotalBought', 'u32TotalSold', 'fBuyPrice', 'fSellPrice', 'fTemperature_avg']
    cols_to_keep = currents + date_cols + other_cols
    return data[cols_to_keep]


def add_date_features(data):
    data['date'] = data['datetime'].dt.date
    data['year'] = data['datetime'].dt.year
    data['month'] = data['datetime'].dt.month
    data['week'] = data['datetime'].dt.week
    data['weekday'] = data['datetime'].dt.weekday
    data['day'] = data['datetime'].dt.day
    data['hour'] = data['datetime'].dt.hour
    return data


def remove_outliers_lambda(data, percentiles=[5, 95]):
    p0, p1 = np.percentile(data, percentiles)
    data.loc[(data > (p1 + 2*(p1 - p0))) | (data < (p0 - 2*(p1 - p0)))] = np.nan
    return data


def group_data_hourly(data):
    data_hourly = data.groupby(['solbox_id', 'year', 'month', 'day', 'hour']).mean().reset_index()
    data_hourly['datetime'] = data_hourly.apply(lambda x: datetime.datetime(int(x['year']), int(x['month']), int(x['day']), int(x['hour'])), axis=1)
    data_hourly = data_hourly.set_index(['solbox_id', 'datetime'])
    return data_hourly

def add_offset(data):
    if data.min() < 0:
        data += np.abs(data.min())
        
    return data
