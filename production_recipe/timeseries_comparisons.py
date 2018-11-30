import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from production_recipe.helpers import get_recipe_df
import matplotlib.colors as mcolors

recipe_directory = '/Volumes/slim/2017_ProductionRecipes/'
dataset_dir = recipe_directory + '/data/'
results_dir = recipe_directory + '/results/'

df, header = get_recipe_df(dataset_dir)

# cleaning
print('Max Aij: ' + str(df['a'].max()) + ', min Aij: ' + str(df['a'].min()))
df = df[df.a <= 1]
df = df[df.a >= 0]
df = df[df.margin == 1]  # basic prices only
print('Post cleaning records: ' + str(len(df)))

# One hot encode categorical variables
df2 = pd.get_dummies(df, columns=['country', 'margin'])

# Convert to arrays
y = np.array(df2['a'])
X = np.array(df2.drop('a', axis=1))

# Build model
print('Building model...')
regressor = RandomForestRegressor(max_depth=None, min_samples_leaf=1, max_features='auto'
                                  , n_estimators=100, min_samples_split=5)
regressor.fit(X, y)

# Prediction countries
countries = [{'name': 'AUS', 'root_number': 16, 'colour': 'turquoise'}
             , {'name': 'CHN', 'root_number': 37, 'colour': 'mediumpurple'}
             , {'name': 'GRC', 'root_number': 78, 'colour': 'palegreen'}
             , {'name': 'JPN', 'root_number': 98, 'colour': 'indianred'}
             , {'name': 'MEX', 'root_number': 125, 'colour': 'cornflowerblue'}]

# Prediction linkages (C25)
pl_i = 3  # Mining and quarrying
pl_j = 13  # Electricity, Gas and Water

all_years = df['year'].unique()

# Create timeseries plot
legend_labels = []
plt.figure()
for c in countries:

    # Known values
    df_known = df.loc[(df['country'] == c['root_number']) & (df['i'] == pl_i) & (df['j'] == pl_j)]
    plt.scatter(df_known['year'].values, df_known['a'].values, c=list(mcolors.to_rgba(c['colour'])))
    legend_labels.append(c['name'] + '-actual')

    # Get encodings for this country
    df_known_enc = df2.loc[(df2['country_' + str(c['root_number']) + '.0'] == 1) & (df2['i'] == pl_i)
                           & (df2['j'] == pl_j)]
    df_first = df_known_enc.iloc[0]

    # figure out which years to forecast
    existing_years = df_known_enc['year'].values
    forecast_years = [item for item in all_years if item not in existing_years]

    # replicate
    df_replicate = pd.concat([df_known_enc.head(1)] * len(forecast_years), ignore_index=True)
    df_replicate['year'] = forecast_years

    X_test = np.array(df_replicate.drop('a', axis=1))
    y_pred = regressor.predict(X_test)
    plt.scatter(forecast_years, y_pred, c=list(mcolors.to_rgba(c['colour'])), alpha=0.4)
    legend_labels.append(c['name'] + '-predict')

plt.ylabel('aij', fontsize=10, labelpad=10)
plt.ylim(bottom=-0.02)
plt.legend(legend_labels, loc='upper center', bbox_to_anchor=(1.25, 0.8))
plt.savefig(recipe_directory + 'results/scatter_timeseries_predictions.png', dpi=700, bbox_inches='tight')
plt.clf()

