import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from production_recipe.helpers import get_recipe_df, perform_cleaning
import matplotlib.colors as mcolors

recipe_directory = '/Volumes/slim/2017_ProductionRecipes/'
dataset_dir = recipe_directory + '/data/'
results_dir = recipe_directory + '/results/'

df, header, year_labels = get_recipe_df(dataset_dir, 25)

df = perform_cleaning(df)

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
             , {'name': 'MEX', 'root_number': 125, 'colour': 'cornflowerblue'}
             , {'name': 'BRA', 'root_number': 28, 'colour': 'darkorange'}]

# Make year labels
y_labels = []
y_tick_marks = []
counter = 0
step = 3
for idx, item in enumerate(year_labels):
    if counter == step:
        y_tick_marks.append(idx)
        y_labels.append('{0:g}'.format(item))
        if counter == step:
            counter = 0
        else:
            counter = counter + 1
    else:
        counter = counter + 1

# Prediction linkages (C25)
# 1: Mining and quarrying into Electricity, Gas and Water
# 2: Recycling into Other manufacturing
linkages = [{'i': 3, 'j': 13}, {'i': 12, 'j': 11}]

all_years = df['year'].unique()
legend_written = 0

for l in linkages:

    l_i = l['i']
    l_j = l['j']

    legend_labels = []
    plt.figure()
    for c in countries:

        # Known values
        df_known = df.loc[(df['country'] == c['root_number']) & (df['i'] == l_i) & (df['j'] == l_j)]
        plt.scatter(df_known['year'].values, df_known['a'].values, c=list(mcolors.to_rgba(c['colour']))
                    , edgecolors='dimgrey', alpha=0.9)
        legend_labels.append(c['name'] + '-actual')

        # Get encodings for this country
        df_known_enc = df2.loc[(df2['country_' + str(c['root_number']) + '.0'] == 1) & (df2['i'] == l_i)
                               & (df2['j'] == l_j)]
        df_first = df_known_enc.iloc[0]

        # figure out which years to forecast
        existing_years = df_known_enc['year'].values
        forecast_years = [item for item in all_years if item not in existing_years]

        # replicate
        df_replicate = pd.concat([df_known_enc.head(1)] * len(forecast_years), ignore_index=True)
        df_replicate['year'] = forecast_years

        X_test = np.array(df_replicate.drop('a', axis=1))
        y_pred = regressor.predict(X_test)
        plt.scatter(forecast_years, y_pred, c=list(mcolors.to_rgba(c['colour'])), alpha=0.5)
        legend_labels.append(c['name'] + '-predict')

    plt.ylabel('aij', fontsize=10, labelpad=10)
    plt.ylim(bottom=-0.02)
    plt.xticks(y_tick_marks, y_labels)
    plt.title('i=' + str(l_i) + ', j=' + str(l_j))
    if legend_written == 0:
        plt.legend(legend_labels, loc='upper center', bbox_to_anchor=(1.2, 0.8), fontsize='small', frameon=False)
        legend_written = 1
    plot_fname = 'scatter_timeseries_predictions_i' + str(l_i) + '_j' + str(l_j) + '.png'
    plt.savefig(recipe_directory + 'results/' + plot_fname, dpi=700, bbox_inches='tight')
    plt.clf()
