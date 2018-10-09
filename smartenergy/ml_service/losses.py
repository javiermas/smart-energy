from sklearn.utils import check_array
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = check_array(y_true), check_array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-15))) * 100
