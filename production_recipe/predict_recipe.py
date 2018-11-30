import h5py
import numpy as np
import pandas as pd
import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from quoll.utils import append_df_to_csv
from production_recipe.helpers import get_recipe_df


recipe_directory = '/Volumes/slim/2017_ProductionRecipes/'
dataset_dir = recipe_directory + '/data/'
results_dir = recipe_directory + '/results/'

df, header = get_recipe_df(dataset_dir)

# Cleaning
print('Max Aij: ' + str(df['a'].max()) + ', min Aij: ' + str(df['a'].min()))
df = df[df.a <= 1]
df = df[df.a >= 0]

# One hot encode categorical variables
df2 = pd.get_dummies(df, columns=['country', 'margin'])  # , 'i', 'j'

# Convert to arrays
y = np.array(df2['a'])
X = np.array(df2.drop('a', axis=1))

# Split training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Scale the continuous data
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

# Create results file
results_filename = results_dir + 'model_performance_summary.csv'
results_header = ['model', 'hyper-parameters', 'mae', 'mse', 'rmse', 'r2', 'explained_variance', 'timestamp']

# Models to run
models_to_build = ['decision_tree', 'ridge_regression', 'linear']
# 'decision_tree', 'random_forest', 'gradient_boost', 'svr', 'ada_boost',
# 'k_nearest_neighbors' takes a long time to compute
# 'kernel_ridge_regression' uses too much memory, (run on a subset?)

# Decision tree model
scaling_opts = [True, False]
max_depth_options = [2, 5, 10, None]
min_samples_leaf_options = [1, 5, 10]
min_samples_split_opts = [2, 5, 10]
max_features_options = ['auto', 'sqrt']
criterion_opts = ['mse']  # , 'mae'
splitter_opts = ['best', 'random']

model_type = 'decision_tree'
if model_type in models_to_build:

    print('Building ' + model_type + ' models')

    for s in scaling_opts:

        if s is True:
            X_train_temp = X_train_scale
            X_test_temp = X_test_scale

            for md in max_depth_options:
                for ms in min_samples_leaf_options:
                    for mss in min_samples_split_opts:
                        for mf in max_features_options:
                            for c in criterion_opts:
                                for sp in splitter_opts:
                                    regressor = DecisionTreeRegressor(max_depth=md, min_samples_leaf=ms, max_features=mf
                                                                      , splitter=sp, criterion=c, min_samples_split=mss)
                                    regressor.fit(X_train_temp, y_train)
                                    y_pred = regressor.predict(X_test_temp)

                                    results = [{'model': model_type, 'hyper-parameters': [md, ms, mf, sp, c, s]
                                                , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                                                , 'mse': metrics.mean_squared_error(y_test, y_pred)
                                                , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                                                , 'r2': metrics.r2_score(y_test, y_pred)
                                                , 'explained_variance': metrics.explained_variance_score(y_test, y_pred)
                                                , 'timestamp': str(datetime.datetime.now())
                                                }]

                                    append_df_to_csv(pd.DataFrame(results, columns=results_header), results_filename, sep=',',
                                                     header=results_header)

# Random forest
model_type = 'random_forest'
max_depth_options = [10, 50, None]  # 5,
n_estimators_options = [50, 100]  # 150, 200
min_samples_split_opts = [2, 5, 10]
if model_type in models_to_build:

    print('Building ' + model_type + ' models')

    for s in scaling_opts:
        if s is not True:
            X_train_temp = X_train
            X_test_temp = X_test

            for md in max_depth_options:
                for ms in min_samples_leaf_options:
                    for mss in min_samples_split_opts:
                        for mf in max_features_options:
                            for ne in n_estimators_options:
                                regressor = RandomForestRegressor(max_depth=md, min_samples_leaf=ms, max_features=mf
                                                                  , n_estimators=ne, min_samples_split=mss)
                                regressor.fit(X_train_temp, y_train)
                                y_pred = regressor.predict(X_test_temp)

                                results = [{'model': model_type, 'hyper-parameters': [md, ms, mf, ne, mss, s]
                                            , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                                            , 'mse': metrics.mean_squared_error(y_test, y_pred)
                                            , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                                            , 'r2': metrics.r2_score(y_test, y_pred)
                                            , 'explained_variance': metrics.explained_variance_score(y_test, y_pred)
                                            , 'timestamp': str(datetime.datetime.now())
                                            }]

                                df = pd.DataFrame(results, columns=results_header)
                                append_df_to_csv(df, results_filename, sep=',', header=results_header)

# Ridge regression
fit_intercept_opts = [True, False]
alpha_opts = [1e-4, 1e-3, 1e-2, 1, 5, 10, 20]
normalise_opts = [True, False]

model_type = 'ridge_regression'
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

        for fi in fit_intercept_opts:
            for a in alpha_opts:
                for nm in normalise_opts:
                    regressor = Ridge(alpha=a, normalize=nm, fit_intercept=fi)
                    regressor.fit(X_train_temp, y_train)
                    y_pred = regressor.predict(X_test_temp)

                    results = [{'model': model_type, 'hyper-parameters': [fi, a, nm, s]
                                , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                                , 'mse': metrics.mean_squared_error(y_test, y_pred)
                                , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                                , 'r2': metrics.r2_score(y_test, y_pred)
                                , 'explained_variance': metrics.explained_variance_score(y_test, y_pred)
                                , 'timestamp': str(datetime.datetime.now())
                                }]

                    df = pd.DataFrame(results, columns=results_header)
                    append_df_to_csv(df, results_filename, sep=',', header=results_header)

# k nearest neighbors
n_neighbors_opts = [2, 5]
weights_opts = ['distance']  # 'uniform',
algorithm_opts = ['auto']  # , 'kd_tree' , 'ball_tree'

model_type = 'k_nearest_neighbors'
if model_type in models_to_build:

    print('Building ' + model_type + ' models')
    results = []

    for s in scaling_opts:
        if s is True:  # do not process unscaled case
            X_train_temp = X_train_scale
            X_test_temp = X_test_scale

            for n in n_neighbors_opts:
                for w in weights_opts:
                    for a in algorithm_opts:
                        regressor = KNeighborsRegressor(n_neighbors=n, weights=w, algorithm=a)
                        regressor.fit(X_train_temp, y_train)
                        y_pred = regressor.predict(X_test_temp)

                        results = [{'model': model_type, 'hyper-parameters': [n, w, a, s]
                                    , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                                    , 'mse': metrics.mean_squared_error(y_test, y_pred)
                                    , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                                    , 'r2': metrics.r2_score(y_test, y_pred)
                                    , 'explained_variance': metrics.explained_variance_score(y_test, y_pred)
                                    , 'timestamp': str(datetime.datetime.now())
                                     }]

                        df = pd.DataFrame(results, columns=results_header)
                        append_df_to_csv(df, results_filename, sep=',', header=results_header)


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

                    results = [{'model': model_type, 'hyper-parameters': [k, g, a, s]
                                , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                                , 'mse': metrics.mean_squared_error(y_test, y_pred)
                                , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                                , 'r2': metrics.r2_score(y_test, y_pred)
                                , 'explained_variance': metrics.explained_variance_score(y_test, y_pred)
                                , 'timestamp': str(datetime.datetime.now())
                                }]

                    df = pd.DataFrame(results, columns=results_header)
                    append_df_to_csv(df, results_filename, sep=',', header=results_header)


# Gradient boost regression
loss_opts = ['ls', 'huber']  # 'lad'
learning_rate_opts = [0.01, 0.1, 0.5, 1]
n_estimators_options = [100, 150, 200]

model_type = 'gradient_boost'
if model_type in models_to_build:

    print('Building ' + model_type + ' models')

    X_train_temp = X_train
    X_test_temp = X_test

    for ne in n_estimators_options:
        for l in loss_opts:
            for lr in learning_rate_opts:
                for mf in max_features_options:
                    regressor = GradientBoostingRegressor(n_estimators=ne, loss=l, learning_rate=lr
                                                          , max_features=mf)
                    regressor.fit(X_train_temp, y_train)
                    y_pred = regressor.predict(X_test_temp)

                    results = [{'model': model_type, 'hyper-parameters': [ne, l, lr, mf]
                                , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                                , 'mse': metrics.mean_squared_error(y_test, y_pred)
                                , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                                , 'r2': metrics.r2_score(y_test, y_pred)
                                , 'explained_variance': metrics.explained_variance_score(y_test, y_pred)
                                , 'timestamp': str(datetime.datetime.now())
                                }]

                    df = pd.DataFrame(results, columns=results_header)
                    append_df_to_csv(df, results_filename, sep=',', header=results_header)


# SVR
kernel_opts = ['linear']  # , 'rbf'
gamma_opts = [0.1, 1, 10]

model_type = 'svr'
if model_type in models_to_build:

    print('Building ' + model_type + ' models')

    results = []

    X_train_temp = X_train_scale
    X_test_temp = X_test_scale

    for k in kernel_opts:
        for g in gamma_opts:
            regressor = SVR(kernel=k, gamma=g)
            regressor.fit(X_train_temp, y_train)
            y_pred = regressor.predict(X_test_temp)

            results = [{'model': model_type, 'hyper-parameters': [k, g]
                        , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                        , 'mse': metrics.mean_squared_error(y_test, y_pred)
                        , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                        , 'r2': metrics.r2_score(y_test, y_pred)
                        , 'explained_variance': metrics.explained_variance_score(y_test, y_pred)
                        , 'timestamp': str(datetime.datetime.now())
                        }]

            df = pd.DataFrame(results, columns=results_header)
            append_df_to_csv(df, results_filename, sep=',', header=results_header)

# Linear model
fit_intercept_opts = [True, False]

model_type = 'linear'
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

        for f in fit_intercept_opts:

            regressor = LinearRegression(fit_intercept=f)
            regressor.fit(X_train_temp, y_train)
            y_pred = regressor.predict(X_test_temp)

            results = [{'model': model_type, 'hyper-parameters': [f, s]
                        , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                        , 'mse': metrics.mean_squared_error(y_test, y_pred)
                        , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                        , 'r2': metrics.r2_score(y_test, y_pred)
                        , 'explained_variance': metrics.explained_variance_score(y_test, y_pred)
                        , 'timestamp': str(datetime.datetime.now())
                        }]

            df = pd.DataFrame(results, columns=results_header)
            append_df_to_csv(df, results_filename, sep=',', header=results_header)


model_type = 'ada_boost'
loss_opts = ['linear', 'square', 'exponential']
min_samples_leaf_opts = [1, 5, 10]

if model_type in models_to_build:

    print('Building ' + model_type + ' models')

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
                    for ms in min_samples_split_opts:
                        regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=None, splitter='best'
                                                                            , criterion='mse', min_samples_leaf=ms)
                                                      , n_estimators=ne, loss=l, learning_rate=lr)

                        regressor.fit(X_train_temp, y_train)
                        y_pred = regressor.predict(X_test_temp)

                        results = [{'model': model_type, 'hyper-parameters': [ne, l, lr, s]
                                    , 'mae': metrics.mean_absolute_error(y_test, y_pred)
                                    , 'mse': metrics.mean_squared_error(y_test, y_pred)
                                    , 'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                                    , 'r2': metrics.r2_score(y_test, y_pred)
                                    , 'explained_variance': metrics.explained_variance_score(y_test, y_pred)
                                    , 'timestamp': str(datetime.datetime.now())
                                    }]

                        append_df_to_csv(pd.DataFrame(results, columns=results_header), results_filename, sep=',',
                                         header=results_header)