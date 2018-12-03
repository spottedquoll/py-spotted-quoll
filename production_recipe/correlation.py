from production_recipe.helpers import get_recipe_df
import pandas as pd
import matplotlib.pyplot as plt

recipe_directory = '/Volumes/slim/2017_ProductionRecipes/'
dataset_dir = recipe_directory + '/data/'

# Read raw data
df, header, year_labels = get_recipe_df(dataset_dir, 25)

# cleaning
print('Max Aij: ' + str(df['a'].max()) + ', min Aij: ' + str(df['a'].min()))
df = df[df.a <= 1]
df = df[df.a >= 0]

pd.plotting.scatter_matrix(df, figsize=(8, 8))
plt.savefig(recipe_directory + 'results/correlation_matrix.png', dpi=700)
plt.clf()

