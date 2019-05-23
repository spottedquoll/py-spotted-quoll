import h5py
import numpy as np

# Paths
work_directory = '/Volumes/slim/Projects/isa_projects/malaria/figure_fix_2/'
processed_data_dir = work_directory + '/processed_data_ml/'

# Processed data
f = h5py.File(processed_data_dir + 'processed_data.h5', 'r')

# Settings
country_groups = 3

for i in range(country_groups ):
    trade = np.array(f['malaria_products_in_trade_' + str(i+1)])
    deforestation = np.array(f['deforestation_' + str(i+1)])
    new_trade = np.array(f['malaria_net_trade_' + str(i+1)])