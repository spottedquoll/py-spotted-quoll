import numpy as np


def mean_absolute_percentage_error(y_actual, y_pred):
    return np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
