import os
import numbers
import numpy as np


def append_df_to_csv(df, filename, header, sep=',', encoding='utf-8'):

    if not os.path.isfile(filename):
        df.to_csv(filename, mode='w', index=False, sep=sep, header=header, encoding=encoding)
    else:
        df.to_csv(filename, mode='a', index=False, sep=sep, header=False, encoding=encoding)


def custom_tick_labels(all_labels, step_size):

    y_labels = []
    y_tick_marks = []
    counter = 0
    for idx, item in enumerate(all_labels):
        if counter == step_size:
            y_tick_marks.append(idx)
            y_labels.append(item)
            if counter == step_size:
                counter = 0
            else:
                counter = counter + 1
        else:
            counter = counter + 1


def mixed_list_to_string(llist, delimiter):

    final_str = ''
    first = True

    for item in llist:
        if first:
            d = ''
            first = False
        else:
            d = delimiter

        if item is None:
            final_str = final_str + d + str('None')
        elif isinstance(item, numbers.Number):
            final_str = final_str + d + str(item)
        else:
            final_str = final_str + d + item

    return final_str


def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def key_match_in_dicts(list_dicts, search_key, search_value):
    matches = []
    for a in list_dicts:
        if search_key in a and a[search_key] == search_value:
            matches.append(a)

    return matches


def get_local_volume():

    if os.path.exists('/Volumes/slim/'):
        volume = '/Volumes/slim/'
    elif os.path.exists('/Volumes/seagate/'):
        volume = '/Volumes/seagate/Projects_backup/'
    else:
        raise ValueError('No external disk found.')

    return volume


def append_array(vals, store):

    if store is None:
        store = np.array(vals)
    elif store.ndim < 2:
        store = np.append([store], [np.array(vals)], axis=0)
    else:
        store = np.append(store, [np.array(vals)], axis=0)

    return store