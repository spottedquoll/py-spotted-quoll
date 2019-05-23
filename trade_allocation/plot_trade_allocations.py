from trade_allocation.spatial import read_shape_files, read_sa2_meta
import h5py
from trade_allocation.lib import get_port_index, colour_polygons_by_vector, collate_weights
from quoll.utils import get_local_volume
import pyexcel as pe
from numpy import genfromtxt
import numpy as np
from os.path import isfile


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
colour_maps = ['plasma']  # , 'Greys', 'hot', 'viridis', 'BuPu', 'PuRd' 'plasma_r', 'Greys', 'Greys_r',
scalings = ['linear']  # , 'symlog' normalisation of figures
aus_bounding_box = [110, 155, -45, -5]  # x_min, x_max, y_min, y_max]
qld_bounding_box = [136, 155, -30, -8]
colour_bar = [False]  # True,
discrete_colours = [15]  # False
sectors = ['beef', 'meat']  # , 'beef'
years = ['2000', '2009']  # '2005',

# Plot maps
for y in years:
    print(y)

    # extract weights for QA
    for flow in trade_directions:
        collate_weights(y, allocations, input_path, flow, port_locations, 'Brisbane', results_path)

    for flow in trade_directions:
        print('Plotting ' + flow)

        for alloc in allocations:
            print('.' + alloc)

            for scale in scalings:
                print('..' + scale)

                for c in colour_maps:
                    print('...' + c)

                    for cb in colour_bar:
                        print('....cbar:' + str(cb))

                        for dc in discrete_colours:
                            print('.....discol:' + str(dc))

                            for s in sectors:
                                print('......' + s)

                                save_fname_base = (results_path + flow + '_' + alloc + '_' + scale + '_' + c
                                                   + '_cbar' + str(cb) + '_' + s + '_discol' + str(dc)
                                                   + '_' + y).lower()

                                # Whole of Aus trade
                                trade_data = genfromtxt(input_path + alloc + '_' + flow + '_' + s
                                                        + '_domestic_flows_aus' + '_' + y + '.csv', delimiter=',')

                                if trade_data.ndim > 1:
                                    trade_data = np.sum(trade_data, axis=1)  # sum over the port dimension

                                save_fname = save_fname_base + '_aus' + '.png'
                                colour_polygons_by_vector(trade_data, all_shapes, aus_sa2s, save_fname
                                                          , bounding_box=aus_bounding_box, normalisation=scale
                                                          , colour_map=c, attach_colorbar=cb, discrete_bins=dc)

                                # Queensland trade
                                #  Only through port of Brisbane
                                region = 'qld'
                                port_name = 'Brisbane'
                                data_fname = (input_path + alloc + '_' + flow + '_' + s + '_domestic_flows_qld'
                                              + '_' + y + '.csv')

                                if isfile(data_fname):
                                    trade_data = genfromtxt(data_fname, delimiter=',')
                                    matching_ports = get_port_index(port_name, port_locations)

                                    #   Squash the totals
                                    a = np.array(trade_data)
                                    region_totals = None
                                    for m in matching_ports:
                                        if region_totals is None:
                                            region_totals = a[:, m]
                                        else:
                                            region_totals = region_totals + a[:, m]

                                    save_fname = save_fname_base + '_' + region + '_' + port_name + '.png'
                                    colour_polygons_by_vector(region_totals, all_shapes, qld_sa2_members, save_fname
                                                              , bounding_box=qld_bounding_box, normalisation=scale
                                                              , colour_map=c, attach_colorbar=cb, discrete_bins=dc)

                                else:
                                    print('Could not find ' + data_fname)

# Plot using same colour scales


print('Finished')