import numpy as np
from scipy.spatial import distance


def mean_absolute_percentage_error(y_actual, y_pred):
    return np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100


def closest_point(node, nodes):

    if node.ndim != 1:
        raise ValueError('Incorrect input dimensions')

    nodes = np.asarray(nodes)

    if nodes.ndim == 1:
        nodes = [nodes]

    dist = distance.cdist([node], nodes)
    min_dist = np.min(dist)

    return min_dist
