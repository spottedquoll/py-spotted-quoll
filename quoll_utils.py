import pandas as pd
import os


def append_df_to_csv(df, filename, header, sep=',', encoding='utf-8'):

    if not os.path.isfile(filename):
        df.to_csv(filename, mode='w', index=False, sep=sep, header=header, encoding=encoding)
    else:
        df.to_csv(filename, mode='a', index=False, sep=sep, encoding=encoding)