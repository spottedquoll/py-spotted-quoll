import h5py
from numpy import array, log10, sign, multiply, concatenate
import csv
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from quoll.discrete_colour_lists import pinky_greens

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

# Axis labels
x_label_str = 'Net trade balance of deforestation-implicated products (log$_{10}$[\$bn])'
y_label_str = 'Cumulative change in forestation since 1995 (log$_{10}$ [''000 km$^2$])'

# Settings
country_groups = 3
the_colours = pinky_greens()
sampling_opts = [1, 3]  # Sampling (plot every nth sample)
line_thickness_scaler = 4

# Process each subset
for i in range(country_groups):

    # Unpack
    malaria_products_trade = array(h5_store['malaria_products_in_trade_' + str(i + 1)])
    deforestation = array(h5_store['deforestation_' + str(i+1)])
    malaria_net_trade = array(h5_store['malaria_net_trade_' + str(i + 1)])
    malaria_net_trade_pre_scaled = array(h5_store['malaria_net_trade_scaled_' + str(i + 1)])

    countries = country_labels[i]

    # Scale
    net_trade_scaled = multiply(sign(malaria_net_trade), log10(abs(malaria_net_trade))) # this normalised between 0 and 1
    # malaria_trade_scaled = log10(abs(malaria_products_trade))z

    # Tests
    assert(len(countries) == malaria_products_trade.shape[1])
    assert(len(countries) == deforestation.shape[1])
    assert(len(countries) == malaria_net_trade.shape[1])
    assert(len(countries) == malaria_net_trade_pre_scaled.shape[1])

    for s in sampling_opts:

        # Create figure object
        fig, ax = plt.subplots()

        for idx, country in enumerate(countries):

            # x, y, z
            x = malaria_products_trade[:, idx]  # trade balance of malaria implicated products
            y = deforestation[:, idx]  # net forestation
            line_thickness = malaria_net_trade_pre_scaled[:, idx] * line_thickness_scaler  # malaria risk trade

            # Sub-sample
            x_sample = x[::s].copy()
            y_sample = y[::s].copy()
            line_thickness_sample = line_thickness[::s].copy()

            # Line segments
            points = array([x_sample, y_sample]).T.reshape(-1, 1, 2)
            segments = concatenate([points[:-1], points[1:]], axis=1)
            ax.add_collection(LineCollection(segments, linewidths=line_thickness_sample
                                             , color='xkcd:' + the_colours[idx]))

        # Axes options
        plt.autoscale(tight=True)
        ax.axhline(0, color='grey', linewidth=0.4, label='_nolegend_')
        ax.axvline(0, color='grey', linewidth=0.4, label='_nolegend_')

        # Labels
        plt.xlabel(x_label_str, fontsize=10, labelpad=15)
        plt.ylabel(y_label_str, fontsize=10, labelpad=15)

        # Legend
        legend = plt.legend(countries, loc='best', fontsize='small', frameon=False)
        #   Make all the legend lines the same thickness
        for line in legend.get_lines():
            line.set_linewidth(2.5)

        # Outer frame colour
        plt.setp(ax.spines.values(), color='grey')
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color='grey')

        # Limits
        plt.ylim(bottom=1.05*deforestation.min(), top=1.05*deforestation.max())
        plt.xlim(left=1.05 * malaria_products_trade.min(), right=1.05 * malaria_products_trade.max())

        # Save
        plot_fname = 'malaria_forestry_trends_c' + str(i) + '_s' + str(s) + '.png'
        fig.savefig(save_dir + plot_fname, dpi=700, bbox_inches='tight')

        # Finished with this figure
        plt.clf()
        plt.close(fig)
