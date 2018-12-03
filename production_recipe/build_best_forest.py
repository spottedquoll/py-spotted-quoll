import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from production_recipe.helpers import get_recipe_df, perform_cleaning

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

# Split training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Build model
print('Building model...')

regressor = RandomForestRegressor(max_depth=None, min_samples_leaf=1, max_features='auto'
                                  , n_estimators=100, min_samples_split=5)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
# mape = mean_absolute_percentage_error(y_test, y_pred)
print('rmse: ' + str(rmse) + ' , r2: ' + str(r2))

plt.figure()
plt.scatter(y_test, y_pred, s=15)
plt.xlabel('Aij (actual)', fontsize=10, labelpad=10)
plt.ylabel('Aij (predicted)', fontsize=10, labelpad=10)
plt.title('Random forest model, RMSE: ' + '{:.3f}'.format(rmse) + ', r2: ' + '{:.3f}'.format(r2), fontsize=8)
plt.savefig(recipe_directory + 'results/scatter_results_forest.png', dpi=700)
plt.clf()
