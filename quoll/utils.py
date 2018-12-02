import os


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