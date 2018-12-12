import h5py
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as colors

print('Building heat maps')

work_directory = '/Volumes/slim/2018_ielab_test_heatmaps/'
phase = '300'
loop = '005'

# harry's table
filename = work_directory + 'converted_hdf/' + 'table_' + phase + '_' + loop + '.h5'
f = h5py.File(filename, 'r')
table = np.array(f['table'])

# Replace zeros
table[table <= 0] = 0.001

plt.figure()
# ax = plt.axes()
# ax.set_aspect('equal')
plt.imshow(table, interpolation='none', norm=colors.LogNorm(np.min(table), np.max(table)), cmap='plasma')
plt.colorbar()
plt.savefig(work_directory + 'heatmaps/' + 'heatmap_' + phase + '_' + loop + '.png', dpi=700)

print('Finished')