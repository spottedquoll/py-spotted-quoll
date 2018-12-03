import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from production_recipe.helpers import get_recipe_df, perform_cleaning

recipe_directory = '/Volumes/slim/2017_ProductionRecipes/'
dataset_dir = recipe_directory + '/data/'
results_dir = recipe_directory + '/results/'

df, header, year_labels = get_recipe_df(dataset_dir, 25)

df = perform_cleaning(df)

# One hot encode categorical variables
df2 = pd.get_dummies(df, columns=['country', 'margin'])

# Filter for Australian table (2009, ba) (one table is not used to train the model) (leave one out)
df_aus = df2.loc[(df2['country_12.0'] == 1) & (df2['year'] == 20)]
df_aus.reset_index()

if len(df_aus) == 0:
    raise ValueError('Australian test set is empty')

# Remove this single AUS table from the training data
df2.drop(df2.loc[(df2['margin_1.0'] == 1) & (df2['country_12.0'] == 1) & (df2['year'] == 20)].index, inplace=True)
df2.reset_index()

# Training
y_train = np.array(df2['a'])
X_train = np.array(df2.drop('a', axis=1))

# Test
y_test = np.array(df_aus['a'])
X_test = np.array(df_aus.drop('a', axis=1))

# Build model
print('Building model...')

regressor = RandomForestRegressor(max_depth=None, min_samples_leaf=1, max_features='auto'
                                  , n_estimators=100, min_samples_split=5)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print('rmse: ' + str(rmse) + ' , r2: ' + str(r2) + ', mape: ' + str(mae))

# rebuild matrices
aus_data = df_aus['a'].values
aus_matrix = aus_data.reshape(25, 25)
aus_actual = np.transpose(aus_matrix)

aus_predict = y_pred.reshape(25, 25)
aus_predict = np.transpose(aus_predict)

# Plot results
plt.figure()
plt.imshow(aus_actual, interpolation='none', cmap='plasma')
plt.colorbar()
plt.title('Actual')
plt.savefig(recipe_directory + 'results/' + 'heatmap_aus_2009_actual.png', dpi=800, bbox_inches='tight')
plt.clf()

plt.figure()
plt.imshow(aus_predict, interpolation='none', cmap='plasma')
plt.colorbar()
plt.title('Predicted')
plt.savefig(recipe_directory + 'results/' + 'heatmap_aus_2009_predict.png', dpi=800, bbox_inches='tight')
plt.clf()

dist = np.abs(aus_actual-aus_predict)
plt.figure()
plt.imshow(dist, interpolation='none', cmap='plasma')
plt.colorbar()
plt.title('Absolute error')
plt.savefig(recipe_directory + 'results/' + 'heatmap_aus_2009_distance_error.png', dpi=800, bbox_inches='tight')
plt.clf()

# Remove all AUS tables from the training data
df3 = df2.copy()
df3.drop(df3.loc[(df3['country_12.0'] == 1)].index, inplace=True)
df3.reset_index()

# Training
y_train = np.array(df3['a'])
X_train = np.array(df3.drop('a', axis=1))

# Build model
print('Building model...')

regressor = RandomForestRegressor(max_depth=None, min_samples_leaf=1, max_features='auto'
                                  , n_estimators=100, min_samples_split=5)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print('rmse: ' + str(rmse) + ' , r2: ' + str(r2) + ', mae: ' + str(mae))

# Make results matrix
aus_predict = y_pred.reshape(25, 25)
aus_predict = np.transpose(aus_predict)

plt.figure()
plt.imshow(aus_predict, interpolation='none', cmap='plasma')
plt.colorbar()
plt.title('Predicted')
plt.savefig(recipe_directory + 'results/' + 'heatmap_aus_2009_predict_no_AUS.png', dpi=800, bbox_inches='tight')
plt.clf()

dist = np.abs(aus_actual-aus_predict)
plt.figure()
plt.imshow(dist, interpolation='none', cmap='plasma')
plt.colorbar()
plt.title('Absolute error')
plt.savefig(recipe_directory + 'results/' + 'heatmap_aus_2009_distance_error_no_AUS.png', dpi=800, bbox_inches='tight')
plt.clf()