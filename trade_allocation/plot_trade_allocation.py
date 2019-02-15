from trade_allocation.spatial import read_shape_files, read_sa2_meta
import h5py
from trade_allocation.lib import plot_qld_beef_exports, plot_aus_beef_exports, plot_qld_beef_exports_single_port\
    , colour_polygons_by_vector, collate_weights
from quoll.utils import get_local_volume
import pyexcel as pe
from numpy import genfromtxt
import numpy as np

volume = get_local_volume()
work_path = volume + 'more_projects/nesting/'
asgs_path = work_path + 'ASGS/SA2/2011/'
input_path = work_path + 'allocation_results/'
results_path = work_path + 'finished_figures/'

# Get SA2 meta information and shape files
sa2_meta_store = read_sa2_meta(asgs_path + 'SA2_2011_AUST.xlsx')
all_shapes = read_shape_files(asgs_path + 'SA2_2011_AUST.shp')

if len(all_shapes) != len(sa2_meta_store):
    raise ValueError('SA2 meta version is different to the shape file version')

# SA2 member lists
#   QLD
filename = input_path + 'state_sa2_members_qld.h5'
f = h5py.File(filename, 'r')
qld_sa2_members = list(f['sa2_members'])
#   Aus
aus_sa2s = list(range(1, 2215))

# Port names and locations
port_locations = []
records = pe.iget_records(file_name=str(work_path) + '/ABS/' + 'Australian_port_locations.xlsx')
for row in records:
    port_locations.append(row)

# Options
allocations = ['production', 'gravity', 'combined']
trade_directions = ['exports']  # 'imports',
colour_maps = ['plasma']  # , 'hot', 'viridis''BuPu', 'PuRd' 'plasma_r', 'Greys', 'Greys_r',
scalings = ['linear', 'log']  # normalisation of figures
aus_bounding_box = [110, 155, -45, -5]

# extract weights for QA
for flow in trade_directions:
    collate_weights(allocations, input_path, flow, port_locations, 'Brisbane', results_path)

# Plot maps
for flow in trade_directions:
    for a in allocations:
        for scale in scalings:

            print('Plotting allocation = ' + str(a))

            for c in colour_maps:

                print('Colour maps = ' + c)

                plot_qld_beef_exports(a, flow, all_shapes, input_path, results_path, qld_sa2_members, normalise=True
                                      , colour_map=c)

                plot_aus_beef_exports(a, flow, all_shapes, input_path, results_path, normalise=False, colour_map=c)

                plot_qld_beef_exports_single_port(a, flow, all_shapes, input_path, results_path, qld_sa2_members
                                                  , 'Brisbane', port_locations, normalise=True, colour_map=c)

                print('Beef exports from whole of Aus')
                trade_data = genfromtxt(input_path + a + '_' + flow + '_beef_domestic_flows_aus.csv', delimiter=',')
                save_file_name = results_path + 'aus_beef_' + flow + '_' + a + '_' + c + '_' + scale + '.png'
                colour_polygons_by_vector(trade_data, all_shapes, aus_sa2s, save_file_name, aus_bounding_box
                                          , normalisation=scale, colour_map=c)

                print('Meat exports from whole of Aus')
                sectors = 'meat'
                trade_data = genfromtxt(input_path + a + '_' + flow + '_' + sectors + '_domestic_flows_aus.csv'
                                        , delimiter=',')
                trade_data = np.sum(trade_data, axis=1)  # sum over the port dimension
                save_file_name = results_path + 'aus_' + sectors + '_' + flow + '_' + a + '_' + c + '_' + scale + '.png'
                colour_polygons_by_vector(trade_data, all_shapes, aus_sa2s, save_file_name, aus_bounding_box
                                          , normalisation=scale, colour_map=c)


print('Finished')