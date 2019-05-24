import h5py
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Notes
# x-axis: trade balance of malaria implicated products
# y-axis: net forestation
# weighting: malaria risk

# Paths
work_directory = '/Volumes/slim/Projects/isa_projects/malaria/figure_fix_2/'
processed_data_dir = work_directory + 'processed_data_ml/'
save_dir = work_directory + 'figs/'

# Processed data
h5_store = h5py.File(processed_data_dir + 'processed_data.h5', 'r')

# Country labels
with open(processed_data_dir + 'country_labels.csv', 'r') as f:
    reader = csv.reader(f)
    country_labels = list(reader)

# List of colours
the_colours = ['bright violet', 'fuchsia', 'pinkish red', 'pastel pink', 'pinkish grey', 'robin egg blue', 'turquoise'
               , 'blue green', 'dark teal', 'forrest green']

# Settings
country_groups = 3

for i in range(country_groups):

    # Figure object
    legend_labels = []
    fig, ax = plt.subplots()

    # Unpack
    malaria_trade = np.array(h5_store['malaria_products_in_trade_' + str(i+1)])
    deforestation = np.array(h5_store['deforestation_' + str(i+1)])
    new_trade = np.array(h5_store['malaria_net_trade_' + str(i+1)])
    countries = country_labels[0]

    # Tests
    assert(len(countries) == malaria_trade.shape[1])
    assert (len(countries) == deforestation.shape[1])
    assert (len(countries) == new_trade.shape[1])

    for idx, country in enumerate(countries):

        # x, y, z
        x = new_trade[:, idx]  # trade balance of malaria implicated products
        y = deforestation[:, idx]  # net forestation
        line_thickness = malaria_trade[:, idx]  # malaria risk trade

        # Line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        ax.add_collection(LineCollection(segments, linewidths=line_thickness, color='xkcd:' + the_colours[idx]))

        # Legend
        legend_labels.append(country)

    # plt.ylabel('aij', fontsize=10, labelpad=10)
    # plt.ylim(bottom=-0.02)
    # plt.xticks(y_tick_marks, y_labels)
    # plt.title('i=' + str(l_i) + ', j=' + str(l_j))
    plt.legend(countries, loc='upper right', fontsize='small', frameon=False)

    plot_fname = 'malaria_forestry_trends_' + str(i) + '.png'
    plt.autoscale(tight=True)
    fig.savefig(save_dir + plot_fname, dpi=700, bbox_inches='tight')
    plt.clf()

# x = new_trade[:, idx]  # trade balance of malaria implicated products
# y = deforestation[:, idx]  # net forestation
# line_thickness = malaria_trade[:, idx]  # malaria risk trade
# points = np.array([x, y]).T.reshape(-1, 1, 2)
# segments = np.concatenate([points[:-1], points[1:]], axis=1)
# lc = LineCollection(segments, linewidths=line_thickness, color='blue')
# fig, a = plt.subplots()
# a.add_collection(lc)