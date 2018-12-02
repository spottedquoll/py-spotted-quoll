import pandas as pd
import h5py
import numpy as np


def get_recipe_df(dataset_dir):

    # Read raw data
    f = h5py.File(dataset_dir + 'a_coefficients_training_set.h5', 'r')
    table = np.array(f['table'])
    table = np.transpose(table)

    print('Read ' + str(table.shape[0]) + ' records')

    # get column headers
    df_header = pd.read_csv(dataset_dir + 'header.csv')
    header = list(df_header.columns)
    header.remove(header[len(header) - 1])

    # Make dataframe
    df = pd.DataFrame(table, columns=header)

    # Years
    year_labels = np.array(f['year_labels'])[0]

    return df, header, year_labels
