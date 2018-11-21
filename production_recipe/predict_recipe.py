import h5py
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge

work_directory = '/Volumes/slim/2017_ProductionRecipes/'

# Read raw data
f = h5py.File(work_directory + 'Results/' + 'a_coefficients_training_set.h5', 'r')
table = np.array(f['table'])
table = np.transpose(table)

print('Read ' + str(table.shape[0]) + ' records')

# Make dataframe
header = ['y', 'i', 'j', 'country', 'year', 'margin']
df = pd.DataFrame(table, columns=header)

# cleaning
print('Max Aij: ' + str(df['y'].max()) + ', min Aij: ' + str(df['y'].min()))
df = df[df.y <= 1]
df = df[df.y >= 0]

# One hot encode categorical variables
df2 = pd.get_dummies(df, columns=['country', 'margin'])

# Convert to arrays
y = np.array(df2['y'])
X = np.array(df2.drop('y', axis=1))

# Split training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

results = []

# Decision tree model
model_type = 'decision_tree'
print('Building ' + model_type + ' models')

max_depth_options = [2, 5, 10, None]
min_samples_leaf_options = [1, 5, 10]
max_features_options = ['auto', 'sqrt']

for md in max_depth_options:
    for ms in min_samples_leaf_options:
        for mf in max_features_options:
            regressor = DecisionTreeRegressor(max_depth=md, min_samples_leaf=ms, max_features=mf)
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)

            results.append({'model': model_type, 'hyper-parameters': [md, ms, mf]
                            , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                            , 'mse': metrics.mean_squared_error(y_test, y_pred)
                            , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                            , 'r2': metrics.r2_score(y_test, y_pred)
                            })

# random forest
model_type = 'random_forest'
print('Building ' + model_type + ' models')

n_estimators_options = [50, 100]  # 200
for md in max_depth_options:
    for ms in min_samples_leaf_options:
        for mf in max_features_options:
            for ne in n_estimators_options:
                regressor = RandomForestRegressor(max_depth=md, min_samples_leaf=ms, max_features=mf, n_estimators=ne)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)

                results.append({'model': model_type, 'hyper-parameters': [md, ms, mf, ne]
                                , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                                , 'mse': metrics.mean_squared_error(y_test, y_pred)
                                , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                                , 'r2': metrics.r2_score(y_test, y_pred)
                                })

# ridge regression
fit_intercept_opts = [True, False]
alpha_opts = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]
normalise_opts = [True, False]

model_type = 'ridge_regression'
print('Building ' + model_type + ' models')

for fi in fit_intercept_opts:
    for a in alpha_opts:
        for nm in normalise_opts:
            regressor = Ridge(alpha=a, normalize=nm, fit_intercept=fi)
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)

            results.append({'model': model_type, 'hyper-parameters': [fi, a, nm]
                            , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                            , 'mse': metrics.mean_squared_error(y_test, y_pred)
                            , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                            , 'r2': metrics.r2_score(y_test, y_pred)
                            })

# k nearest neighbors
n_neighbors_opts = [2, 5, 10]
weights_opts = ['uniform', 'distance']
algorithm_opts = ['auto', 'ball_tree', 'kd_tree']

model_type = 'k_nearest_neighbors'
print('Building ' + model_type + ' models')

for n in n_neighbors_opts:
    for w in weights_opts:
        for a in algorithm_opts:
            regressor = KNeighborsRegressor(n_neighbors=n, weights=w, algorithm=a)
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)

            results.append({'model': model_type, 'hyper-parameters': [n, w, a]
                            , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                            , 'mse': metrics.mean_squared_error(y_test, y_pred)
                            , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                            , 'r2': metrics.r2_score(y_test, y_pred)
                            })

# kernel ridge regression
kernel_opts = ['linear', 'poly', 'rbf']
gamma_opts = [None, 0.1, 1, 10]
alpha_opts = [1e-4, 1e-3, 1e-2, 1, 5, 10]

model_type = 'kernel_ridge_regression'
print('Building ' + model_type + ' models')

for k in kernel_opts:
    for g in gamma_opts:
        for a in alpha_opts:
            regressor = KernelRidge(kernel=k, gamma=g, alpha=a)
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)

            results.append({'model': model_type, 'hyper-parameters': [k, g, a]
                            , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                            , 'mse': metrics.mean_squared_error(y_test, y_pred)
                            , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                            , 'r2': metrics.r2_score(y_test, y_pred)
                            })


# Save results
result_df = pd.DataFrame(results, columns=['model', 'options', 'mae', 'mse', 'rmse', 'r2'])
result_df.to_csv(work_directory + '/results/' + 'summary_results.csv', encoding='utf-8')