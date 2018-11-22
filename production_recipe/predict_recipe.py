import h5py
import numpy as np
import pandas as pd
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import MinMaxScaler
from quoll_utils import append_df_to_csv

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

# Scale the continuous data
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

# Create results file
append_to_existing = 0
results_filename = work_directory + '/results/' + 'summary_results.csv'

# If this is a new run, delete the existing file
if append_to_existing == 1 and os.path.isfile(results_filename):
    os.remove(results_filename)

results_header = ['model', 'options', 'mae', 'mse', 'rmse', 'r2', 'explained_variance']
models_to_build = ['decision_tree', 'random_forest', 'ridge_regression', 'k_nearest_neighbors'
                   , 'kernel_ridge_regression', 'gradient_boost']

# Decision tree model
scaling_opts = [True, False]
max_depth_options = [2, 5, 10, None]
min_samples_leaf_options = [1, 5, 10]
max_features_options = ['auto', 'sqrt']

model_type = 'decision_tree'
if model_type in models_to_build:

    print('Building ' + model_type + ' models')
    results = []

    for s in scaling_opts:
        if s is True:
            X_train_temp = X_train_scale
            X_test_temp = X_test_scale
        else:
            X_train_temp = X_train
            X_test_temp = X_test

        for md in max_depth_options:
            for ms in min_samples_leaf_options:
                for mf in max_features_options:
                    regressor = DecisionTreeRegressor(max_depth=md, min_samples_leaf=ms, max_features=mf)
                    regressor.fit(X_train_temp, y_train)
                    y_pred = regressor.predict(X_test_temp)

                    results.append({'model': model_type, 'hyper-parameters': [md, ms, mf, s]
                                    , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                                    , 'mse': metrics.mean_squared_error(y_test, y_pred)
                                    , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                                    , 'r2': metrics.r2_score(y_test, y_pred)
                                    , 'expl_var': metrics.explained_variance_score(y_test, y_pred)
                                    })

    append_df_to_csv(pd.DataFrame(results), results_filename, sep=',', header=results_header)

# Random forest
model_type = 'random_forest'
n_estimators_options = [50, 100]  # 200
if model_type in models_to_build:

    print('Building ' + model_type + ' models')

    results = []

    for s in scaling_opts:
        if s is True:
            X_train_temp = X_train_scale
            X_test_temp = X_test_scale
        else:
            X_train_temp = X_train
            X_test_temp = X_test

        for md in max_depth_options:
            for ms in min_samples_leaf_options:
                for mf in max_features_options:
                    for ne in n_estimators_options:
                        regressor = RandomForestRegressor(max_depth=md, min_samples_leaf=ms, max_features=mf, n_estimators=ne)
                        regressor.fit(X_train_temp, y_train)
                        y_pred = regressor.predict(X_test_temp)

                        results.append({'model': model_type, 'hyper-parameters': [md, ms, mf, ne, s]
                                        , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                                        , 'mse': metrics.mean_squared_error(y_test, y_pred)
                                        , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                                        , 'r2': metrics.r2_score(y_test, y_pred)
                                        , 'expl_var': metrics.explained_variance_score(y_test, y_pred)
                                        })

    append_df_to_csv(pd.DataFrame(results), results_filename, sep=',', header=results_header)

# Ridge regression
fit_intercept_opts = [True, False]
alpha_opts = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]
normalise_opts = [True, False]

if model_type in models_to_build:

    model_type = 'ridge_regression'
    print('Building ' + model_type + ' models')
    results = []

    for s in scaling_opts:
        if s is True:
            X_train_temp = X_train_scale
            X_test_temp = X_test_scale
        else:
            X_train_temp = X_train
            X_test_temp = X_test

        for fi in fit_intercept_opts:
            for a in alpha_opts:
                for nm in normalise_opts:
                    regressor = Ridge(alpha=a, normalize=nm, fit_intercept=fi)
                    regressor.fit(X_train_temp, y_train)
                    y_pred = regressor.predict(X_test_temp)

                    results.append({'model': model_type, 'hyper-parameters': [fi, a, nm, s]
                                    , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                                    , 'mse': metrics.mean_squared_error(y_test, y_pred)
                                    , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                                    , 'r2': metrics.r2_score(y_test, y_pred)
                                    , 'expl_var': metrics.explained_variance_score(y_test, y_pred)
                                    })
    append_df_to_csv(pd.DataFrame(results), results_filename, sep=',', header=results_header)

# k nearest neighbors
n_neighbors_opts = [2, 5, 10]
weights_opts = ['uniform', 'distance']
algorithm_opts = ['auto', 'ball_tree', 'kd_tree']

model_type = 'k_nearest_neighbors'
if model_type in models_to_build:

    print('Building ' + model_type + ' models')
    results = []

    for s in scaling_opts:
        if s is True:
            X_train_temp = X_train_scale
            X_test_temp = X_test_scale
        else:
            X_train_temp = X_train
            X_test_temp = X_test

        for n in n_neighbors_opts:
            for w in weights_opts:
                for a in algorithm_opts:
                    regressor = KNeighborsRegressor(n_neighbors=n, weights=w, algorithm=a)
                    regressor.fit(X_train_temp, y_train)
                    y_pred = regressor.predict(X_test_temp)

                    results.append({'model': model_type, 'hyper-parameters': [n, w, a, s]
                                    , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                                    , 'mse': metrics.mean_squared_error(y_test, y_pred)
                                    , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                                    , 'r2': metrics.r2_score(y_test, y_pred)
                                    , 'expl_var': metrics.explained_variance_score(y_test, y_pred)
                                    })
    append_df_to_csv(pd.DataFrame(results), results_filename, sep=',', header=results_header)

# Kernel ridge regression
kernel_opts = ['linear', 'poly', 'rbf']
gamma_opts = [None, 0.1, 1, 10]
alpha_opts = [1e-4, 1e-3, 1e-2, 1, 5, 10]

model_type = 'kernel_ridge_regression'
if model_type in models_to_build:

    print('Building ' + model_type + ' models')
    results = []

    for s in scaling_opts:
        if s is True:
            X_train_temp = X_train_scale
            X_test_temp = X_test_scale
        else:
            X_train_temp = X_train
            X_test_temp = X_test

        for k in kernel_opts:
            for g in gamma_opts:
                for a in alpha_opts:
                    regressor = KernelRidge(kernel=k, gamma=g, alpha=a)
                    regressor.fit(X_train_temp, y_train)
                    y_pred = regressor.predict(X_test_temp)

                    results.append({'model': model_type, 'hyper-parameters': [k, g, a, s]
                                    , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                                    , 'mse': metrics.mean_squared_error(y_test, y_pred)
                                    , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                                    , 'r2': metrics.r2_score(y_test, y_pred)
                                    , 'expl_var': metrics.explained_variance_score(y_test, y_pred)
                                    })

    append_df_to_csv(pd.DataFrame(results), results_filename, sep=',', header=results_header)

# Gradient boost regression
loss_opts = ['ls', 'lad']
learning_rate_opts = [0.1, 0.5, 1]

model_type = 'gradient_boost'
if model_type in models_to_build:

    print('Building ' + model_type + ' models')

    results = []

    for s in scaling_opts:
        if s is True:
            X_train_temp = X_train_scale
            X_test_temp = X_test_scale
        else:
            X_train_temp = X_train
            X_test_temp = X_test

            for ne in n_estimators_options:
                for l in loss_opts:
                    for lr in learning_rate_opts:
                        for mf in max_features_options:
                            regressor = GradientBoostingRegressor(n_estimators=ne, loss=l, learning_rate=lr, max_features=mf)
                            regressor.fit(X_train_temp, y_train)
                            y_pred = regressor.predict(X_test_temp)

                            results.append({'model': model_type, 'hyper-parameters': [ne, l, lr, mf, s]
                                            , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                                            , 'mse': metrics.mean_squared_error(y_test, y_pred)
                                            , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                                            , 'r2': metrics.r2_score(y_test, y_pred)
                                            , 'expl_var': metrics.explained_variance_score(y_test, y_pred)
                                            })

    append_df_to_csv(pd.DataFrame(results), results_filename, sep=',', header=results_header)