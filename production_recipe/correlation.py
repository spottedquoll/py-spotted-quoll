import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

recipe_directory = '/Volumes/slim/2017_ProductionRecipes/'

# Read raw data
f = h5py.File(recipe_directory + 'Results/' + 'a_coefficients_training_set.h5', 'r')
table = np.array(f['table'])
table = np.transpose(table)

print('Read ' + str(table.shape[0]) + ' records')

# Make dataframe
header = ['a', 'i', 'j', 'country', 'year', 'margin']
df = pd.DataFrame(table, columns=header)

# cleaning
print('Max Aij: ' + str(df['a'].max()) + ', min Aij: ' + str(df['a'].min()))
df = df[df.a <= 1]
df = df[df.a >= 0]

pd.plotting.scatter_matrix(df, figsize=(8, 8))
plt.savefig(recipe_directory + 'results/correlation_matrix.png', dpi=700)
plt.clf()

