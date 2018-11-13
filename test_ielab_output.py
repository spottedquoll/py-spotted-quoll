import h5py
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as colors

work_directory = '/Users/jacobfry/Desktop/Projects/uni_projects/kleyer_test/'

# harry's table
filename = work_directory + 'table_441.h5'
f = h5py.File(filename, 'r')
table = np.array(f['table'])

# Replace zeros
table[table <= 0] = 0.001

plt.figure()
# ax = plt.axes()
# ax.set_aspect('equal')
plt.imshow(table, interpolation='none', norm=colors.LogNorm(np.min(table), np.max(table)), cmap='plasma')
plt.colorbar()
plt.savefig(work_directory + 'heatmap_001_441.png', dpi=700)
plt.clf()

# basic IE table
filename = work_directory + 'table_060.h5'
f = h5py.File(filename, 'r')
table = np.array(f['table'])

# Replace zeros
table[table <= 0] = 0.001

plt.figure()
plt.imshow(table, interpolation='none', norm=colors.LogNorm(np.min(table), np.max(table)), cmap='plasma')
plt.colorbar()
plt.savefig(work_directory + 'heatmap_001_060.png', dpi=700)
plt.clf()

# SLQ with Harry's constraints
filename = work_directory + 'table_080.h5'
f = h5py.File(filename, 'r')
table = np.array(f['table'])

# Replace zeros
table[table <= 0] = 0.001

plt.figure()
plt.imshow(table, interpolation='none', norm=colors.LogNorm(np.min(table), np.max(table)), cmap='plasma')
plt.colorbar()
plt.savefig(work_directory + 'heatmap_001_080.png', dpi=700)
plt.clf()

# CHARM fix
filename = work_directory + 'table_468.h5'
f = h5py.File(filename, 'r')
table = np.array(f['table'])

# Replace zeros
table[table <= 0] = 0.001

plt.figure()
plt.imshow(table, interpolation='none', norm=colors.LogNorm(np.min(table), np.max(table)), cmap='plasma')
plt.colorbar()
plt.savefig(work_directory + 'heatmap_001_468.png', dpi=700)
plt.clf()