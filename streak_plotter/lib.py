import numpy as np


def symmetric_log(num_list):

    result = []

    for i in num_list:

        if i > 0:
            result.append(np.log10(i))
        elif i < 0:
            result.append(-1*np.log10(-1*i))
        else:
            result.append(i)

    return result
