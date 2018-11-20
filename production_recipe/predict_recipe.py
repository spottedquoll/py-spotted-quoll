import h5py
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics

work_directory = '/Volumes/slim/2017_ProductionRecipes/'

# Read raw data
f = h5py.File(work_directory + 'Results/' + 'a_coefficients_training_set.h5', 'r')
table = np.array(f['table'])
table = np.transpose(table)

print('Read ' + str(table.shape[0]) + ' records')

# Make dataframe
header = ['y', 'i', 'j', 'country', 'year', 'margin']
df = pd.DataFrame(table, columns=header)

print('Max Aij: ' + str(df['y'].max()) + ', min Aij: ' + str(df['y'].min()))

# One hot encode categorical variables
df2 = pd.get_dummies(df, columns=['country', 'margin'])

# Convert to arrays
y = np.array(df2['y'])
X = np.array(df2.drop('y', axis=1))

# Split training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

max_depth_options = [2, 5, 10]
min_samples_leaf_options = [1, 5, 10]
max_features_options = ['auto', 'sqrt']

for md in max_depth_options:
    for ms in min_samples_leaf_options:
        for mf in max_features_options:
            regressor = DecisionTreeRegressor(max_depth=md, min_samples_leaf=ms, max_features=mf)
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)

            print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
            print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
            print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
            print('.')

# RandomForestRegressor(max_features=2, min_samples_split=4, n_estimators=50, min_samples_leaf=2)
# LinearRegression,Ridge,KNeighborsRegressor,DecisionTreeRegressor,RandomForestRegressor, GradientBoostingRegressor