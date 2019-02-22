import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from quoll.metrics import closest_point
from scipy.spatial import distance


def discrete_cmap(n, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # By Jake VanderPlas
    # License: BSD-style

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, n))
    cmap_name = base.name + str(n)

    return LinearSegmentedColormap.from_list(cmap_name, color_list, n)


def find_unique_label_position(x, y, existing_points, closest_allowed, perturb, alter=['x', 'y'], max_attempts=10):

    x_up_test = x
    y_up_test = y
    up_dist = np.min(distance.cdist([np.array([x_up_test, y_up_test])], existing_points))

    x_down_test = x
    y_down_test = y
    down_dist = np.min(distance.cdist([np.array([x_down_test, y_down_test])], existing_points))

    i = 1
    while i <= max_attempts and up_dist <= closest_allowed and down_dist <= closest_allowed:

        # Check up perturbation
        if 'y' in alter:
            y_up_test = y_up_test + i * perturb

        if 'x' in alter:
            x_up_test = x_up_test + i * perturb

        up_dist = np.min(distance.cdist([np.array([x_up_test, y_up_test])], existing_points))

        # Check down perturbation
        if 'y' in alter:
            y_down_test = y_down_test - i * perturb

        if 'x' in alter:
            x_down_test = x_down_test - i * perturb

        down_dist = np.min(distance.cdist([np.array([x_down_test, y_down_test])], existing_points))

        i += 1

    if up_dist > down_dist:
        return x_up_test, y_up_test
    else:
        return x_down_test, y_down_test
