import pandas as pd
import h5py
import numpy as np


def get_recipe_df(dataset_dir, sector_dim):

    # Read raw data
    f = h5py.File(dataset_dir + 'a_coefficients_training_set_' + str(sector_dim) + '.h5', 'r')
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


def perform_cleaning(df):

    print('Max Aij: ' + str(df['a'].max()) + ', min Aij: ' + str(df['a'].min()))

    df = df[df.a <= 1]
    df = df[df.a >= 0]
    df = df[df.margin == 1]  # basic prices only

    print('Post cleaning records: ' + str(len(df)))

    return df
