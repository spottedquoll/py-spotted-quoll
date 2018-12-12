import numpy as np
import pandas as pd
import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from quoll.utils import append_df_to_csv
from production_recipe.helpers import get_recipe_df, perform_cleaning
from quoll.learn import bag_ensemble, ensemble_predict

recipe_directory = '/Volumes/slim/2017_ProductionRecipes/'
dataset_dir = recipe_directory + '/data/'
results_dir = recipe_directory + '/results/'

# Create results file
all_results_filename = results_dir + 'model_performance_summary.csv'
results_header = ['model', 'hyper-parameters', 'mae', 'mse', 'rmse', 'r2', 'explained_variance', 'timestamp']
best_results_filename = results_dir + 'best_in_class_model_performance.csv'

df, header, year_labels = get_recipe_df(dataset_dir, 25)
df = perform_cleaning(df)

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

# Models to run
models_to_build = ['k_nearest_neighbors_bagged']

# 'decision_tree', 'gradient_boost', 'mpl', 'ada_boost'
# 'ridge_regression', 'linear', 'decision_tree', 'random_forest',
# , 'svr'
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
    best_in_class = []

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

                                    results = {'model': model_type, 'hyper-parameters': [md, ms, mf, sp, c, mss, s]
                                                , 'mae': mean_absolute_error(y_test, y_pred)
                                                , 'mse': mean_squared_error(y_test, y_pred)
                                                , 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                                                , 'r2': r2_score(y_test, y_pred)
                                                , 'explained_variance': explained_variance_score(y_test, y_pred)
                                                , 'timestamp': str(datetime.datetime.now())
                                               }

                                    append_df_to_csv(pd.DataFrame([results], columns=results_header)
                                                     , all_results_filename, sep=',', header=results_header)

                                    if best_in_class == [] or best_in_class['rmse'] > results['rmse']:
                                        best_in_class = results

    append_df_to_csv(pd.DataFrame([best_in_class], columns=results_header), best_results_filename, sep=',',
                     header=results_header)

# Random forest
model_type = 'random_forest'
max_depth_options = [10, 50, None]  # 5,
n_estimators_options = [50, 100]  # 150, 200
min_samples_split_opts = [2, 5, 10]
if model_type in models_to_build:

    print('Building ' + model_type + ' models')
    best_in_class = []

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

                                results = {'model': model_type, 'hyper-parameters': [md, ms, mf, ne, mss, s]
                                            , 'mae': mean_absolute_error(y_test, y_pred)
                                            , 'mse': mean_squared_error(y_test, y_pred)
                                            , 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                                            , 'r2': r2_score(y_test, y_pred)
                                            , 'explained_variance': explained_variance_score(y_test, y_pred)
                                            , 'timestamp': str(datetime.datetime.now())
                                            }

                                df = pd.DataFrame([results], columns=results_header)
                                append_df_to_csv(df, all_results_filename, sep=',', header=results_header)

                                if best_in_class == [] or best_in_class['rmse'] > results['rmse']:
                                    best_in_class = results

    append_df_to_csv(pd.DataFrame([best_in_class], columns=results_header),
                     best_results_filename, sep=',',
                     header=results_header)

# Ridge regression
fit_intercept_opts = [True, False]
alpha_opts = [1e-4, 1e-3, 1e-2, 1, 5, 10, 20]
normalise_opts = [True, False]

model_type = 'ridge_regression'
if model_type in models_to_build:

    print('Building ' + model_type + ' models')
    best_in_class = []

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
                    regressor = Ridge(fit_intercept=fi, alpha=a, normalize=nm)
                    regressor.fit(X_train_temp, y_train)
                    y_pred = regressor.predict(X_test_temp)

                    results = {'model': model_type, 'hyper-parameters': [fi, a, nm, s]
                                , 'mae': mean_absolute_error(y_test, y_pred)
                                , 'mse': mean_squared_error(y_test, y_pred)
                                , 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                                , 'r2': r2_score(y_test, y_pred)
                                , 'explained_variance': explained_variance_score(y_test, y_pred)
                                , 'timestamp': str(datetime.datetime.now())
                                }

                    df = pd.DataFrame([results], columns=results_header)
                    append_df_to_csv(df, all_results_filename, sep=',', header=results_header)

                    if best_in_class == [] or best_in_class['rmse'] > results['rmse']:
                        best_in_class = results

    append_df_to_csv(pd.DataFrame([best_in_class], columns=results_header),
                     best_results_filename, sep=',',
                     header=results_header)

# k nearest neighbors
n_neighbors_opts = [2, 5]
weights_opts = ['distance']  # 'uniform',
algorithm_opts = ['auto']  # , 'kd_tree' , 'ball_tree'

model_type = 'k_nearest_neighbors'
if model_type in models_to_build:

    print('Building ' + model_type + ' models')
    best_in_class = []

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

                        results = {'model': model_type, 'hyper-parameters': [n, w, a, s]
                                   , 'mae': mean_absolute_error(y_test, y_pred)
                                   , 'mse': mean_squared_error(y_test, y_pred)
                                   , 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                                   , 'r2': r2_score(y_test, y_pred)
                                   , 'explained_variance': explained_variance_score(y_test, y_pred)
                                   , 'timestamp': str(datetime.datetime.now())
                                   }

                        df = pd.DataFrame([results], columns=results_header)
                        append_df_to_csv(df, all_results_filename, sep=',', header=results_header)

                        if best_in_class == [] or best_in_class['rmse'] > results['rmse']:
                            best_in_class = results

    append_df_to_csv(pd.DataFrame([best_in_class], columns=results_header),
                     best_results_filename, sep=',',
                     header=results_header)


# Kernel ridge regression
kernel_opts = ['linear', 'poly', 'rbf']
gamma_opts = [None, 0.1, 1, 10]
alpha_opts = [1e-4, 1e-3, 1e-2, 1, 5, 10]

model_type = 'kernel_ridge_regression'
if model_type in models_to_build:

    print('Building ' + model_type + ' models')
    best_in_class = []

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

                    results = {'model': model_type, 'hyper-parameters': [k, g, a, s]
                                , 'mae': mean_absolute_error(y_test, y_pred)
                                , 'mse': mean_squared_error(y_test, y_pred)
                                , 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                                , 'r2': r2_score(y_test, y_pred)
                                , 'explained_variance': explained_variance_score(y_test, y_pred)
                                , 'timestamp': str(datetime.datetime.now())
                                }

                    df = pd.DataFrame([results], columns=results_header)
                    append_df_to_csv(df, all_results_filename, sep=',', header=results_header)

                    if best_in_class == [] or best_in_class['rmse'] > results['rmse']:
                        best_in_class = results

    append_df_to_csv(pd.DataFrame([best_in_class], columns=results_header),
                     best_results_filename, sep=',',
                     header=results_header)


# Gradient boost regression
loss_opts = ['ls']  # 'lad', 'huber'
learning_rate_opts = [0.1, 0.5, 1]  # 0.01, 0.1,
n_estimators_options = [200, 225]  # 100, 150,
min_samples_leaf_options = [1, 5]  # , 10
min_samples_split_opts = [2, 5] # , 10
max_features_options = ['auto', 'sqrt']

model_type = 'gradient_boost'
if model_type in models_to_build:

    print('Building ' + model_type + ' models')
    best_in_class = []

    X_train_temp = X_train
    X_test_temp = X_test

    for ne in n_estimators_options:
        for l in loss_opts:
            for lr in learning_rate_opts:
                for mf in max_features_options:
                    for ms in min_samples_leaf_options:
                        for mss in min_samples_split_opts:

                            regressor = GradientBoostingRegressor(n_estimators=ne, loss=l, learning_rate=lr
                                                                  , max_features=mf, min_samples_leaf=ms
                                                                  , min_samples_split=mss)
                            regressor.fit(X_train_temp, y_train)
                            y_pred = regressor.predict(X_test_temp)

                            results = {'model': model_type, 'hyper-parameters': [ne, l, lr, mf, ms, mss]
                                       , 'mae': mean_absolute_error(y_test, y_pred)
                                       , 'mse': mean_squared_error(y_test, y_pred)
                                       , 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                                       , 'r2': r2_score(y_test, y_pred)
                                       , 'explained_variance': explained_variance_score(y_test, y_pred)
                                       , 'timestamp': str(datetime.datetime.now())
                                       }

                            df = pd.DataFrame([results], columns=results_header)
                            append_df_to_csv(df, all_results_filename, sep=',', header=results_header)

                            if best_in_class == [] or best_in_class['rmse'] > results['rmse']:
                                best_in_class = results

    append_df_to_csv(pd.DataFrame([best_in_class], columns=results_header),
                     best_results_filename, sep=',',
                     header=results_header)


# SVR
kernel_opts = ['linear']  # , 'rbf'
gamma_opts = [0.1, 1, 10]

model_type = 'svr'
if model_type in models_to_build:

    print('Building ' + model_type + ' models')
    best_in_class = []

    X_train_temp = X_train_scale
    X_test_temp = X_test_scale

    for k in kernel_opts:
        for g in gamma_opts:
            regressor = SVR(kernel=k, gamma=g)
            regressor.fit(X_train_temp, y_train)
            y_pred = regressor.predict(X_test_temp)

            results = {'model': model_type, 'hyper-parameters': [k, g]
                        , 'mae': mean_absolute_error(y_test, y_pred)
                        , 'mse': mean_squared_error(y_test, y_pred)
                        , 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                        , 'r2': r2_score(y_test, y_pred)
                        , 'explained_variance': explained_variance_score(y_test, y_pred)
                        , 'timestamp': str(datetime.datetime.now())
                       }

            df = pd.DataFrame([results], columns=results_header)
            append_df_to_csv(df, all_results_filename, sep=',', header=results_header)

            if best_in_class == [] or best_in_class['rmse'] > results['rmse']:
                best_in_class = results

    append_df_to_csv(pd.DataFrame([best_in_class], columns=results_header),
                     best_results_filename, sep=',',
                     header=results_header)

# Linear model
fit_intercept_opts = [True, False]

model_type = 'linear'
if model_type in models_to_build:

    print('Building ' + model_type + ' models')
    best_in_class = []

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

            results = {'model': model_type, 'hyper-parameters': [f, s]
                        , 'mae': mean_absolute_error(y_test, y_pred)
                        , 'mse': mean_squared_error(y_test, y_pred)
                        , 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                        , 'r2': r2_score(y_test, y_pred)
                        , 'explained_variance': explained_variance_score(y_test, y_pred)
                        , 'timestamp': str(datetime.datetime.now())
                       }

            df = pd.DataFrame([results], columns=results_header)
            append_df_to_csv(df, all_results_filename, sep=',', header=results_header)

            if best_in_class == [] or best_in_class['rmse'] > results['rmse']:
                best_in_class = results

    append_df_to_csv(pd.DataFrame([best_in_class], columns=results_header),
                     best_results_filename, sep=',',
                     header=results_header)


model_type = 'ada_boost'
loss_opts = ['linear', 'square', 'exponential']
min_samples_leaf_opts = [1, 5, 10]

if model_type in models_to_build:

    print('Building ' + model_type + ' models')
    best_in_class = []

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

                        results = {'model': model_type, 'hyper-parameters': [ne, l, lr, s, ms]
                                    , 'mae': mean_absolute_error(y_test, y_pred)
                                    , 'mse': mean_squared_error(y_test, y_pred)
                                    , 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                                    , 'r2': r2_score(y_test, y_pred)
                                    , 'explained_variance': explained_variance_score(y_test, y_pred)
                                    , 'timestamp': str(datetime.datetime.now())
                                   }

                        append_df_to_csv(pd.DataFrame([results], columns=results_header), all_results_filename, sep=',',
                                         header=results_header)

                        if best_in_class == [] or best_in_class['rmse'] > results['rmse']:
                            best_in_class = results

    append_df_to_csv(pd.DataFrame([best_in_class], columns=results_header),
                     best_results_filename, sep=',',
                     header=results_header)

model_type = 'mpl'
solvers = ['adam']  # , 'lbfgs', 'sgd'
alpha_opts = [1e-6, 1e-5, 1e-4]  # [1e-4, 1e-3, 1e-2, 1, 5, 10]
activation_opts = ['relu']  # 'identity' , 'logistic', 'tanh'
layers = [(df2.shape[1],) * 5, (round(df2.shape[1]*1.5),) * 5, (df2.shape[1],) * 6]  # (df2.shape[1],) * 3,
learning_rate_opts = ['adaptive']  # 'constant', 'invscaling',

if model_type in models_to_build:

    print('Building ' + model_type + ' models')
    best_in_class = []

    for s in scaling_opts:
        if s is True:
            X_train_temp = X_train_scale
            X_test_temp = X_test_scale
        else:
            X_train_temp = X_train
            X_test_temp = X_test

        for a in alpha_opts:
            for sv in solvers:
                for act in activation_opts:
                    for h in layers:
                        for lr in learning_rate_opts:

                            regressor = MLPRegressor(alpha=a, solver=sv, activation=act, hidden_layer_sizes=h
                                                     , learning_rate=lr)

                            regressor.fit(X_train_temp, y_train)
                            y_pred = regressor.predict(X_test_temp)

                            results = {'model': model_type, 'hyper-parameters': [a, sv, act, h, lr, s]
                                       , 'mae': mean_absolute_error(y_test, y_pred)
                                       , 'mse': mean_squared_error(y_test, y_pred)
                                       , 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                                       , 'r2': r2_score(y_test, y_pred)
                                       , 'explained_variance': explained_variance_score(y_test, y_pred)
                                       , 'timestamp': str(datetime.datetime.now())
                                       }

                            append_df_to_csv(pd.DataFrame([results], columns=results_header), all_results_filename, sep=',',
                                             header=results_header)

                            if best_in_class == [] or best_in_class['rmse'] > results['rmse']:
                                best_in_class = results

    append_df_to_csv(pd.DataFrame([best_in_class], columns=results_header),
                     best_results_filename, sep=',',
                     header=results_header)

# k nearest neighbors
n_neighbors_opts = [5, 10]  # 2,
weights_opts = ['distance']  # 'uniform',
algorithm_opts = ['auto']  # , 'kd_tree' , 'ball_tree'
metric_opts = ['cosine', 'jaccard', 'minkowski', 'euclidean']  # , 'euclidean'
folds = [3, 6, 10]
sample_fraction_opts = [1/10, 1/5, 1/4, 1/3]

model_type = 'k_nearest_neighbors_bagged'
if model_type in models_to_build:

    print('Building ' + model_type + ' models')
    best_in_class = []

    training_cols = list(df2.columns)
    del training_cols[0]

    # Post scaling, rejoin y to X to enable sampling
    train_scaled_df = pd.concat(
        [pd.DataFrame(y_train, columns=['a']), pd.DataFrame(X_train_scale, columns=training_cols)]
        , axis=1)
    X_test_temp = X_test_scale

    for n in n_neighbors_opts:
        for w in weights_opts:
            for a in algorithm_opts:
                for f in folds:
                    for fraction in sample_fraction_opts:
                        for m in metric_opts:

                            regressor = KNeighborsRegressor(n_neighbors=n, weights=w, algorithm=a, metric=m)

                            sub_models = bag_ensemble(regressor, train_scaled_df, X_test_temp, y_test
                                                      , y_col_name='a', folds=f, sample_fraction=fraction)

                            y_pred = ensemble_predict(sub_models, X_test_temp)

                            results = {'model': model_type, 'hyper-parameters': [n, w, a, f, fraction, m]
                                       , 'mae': mean_absolute_error(y_test, y_pred)
                                       , 'mse': mean_squared_error(y_test, y_pred)
                                       , 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                                       , 'r2': r2_score(y_test, y_pred)
                                       , 'explained_variance': explained_variance_score(y_test, y_pred)
                                       , 'timestamp': str(datetime.datetime.now())
                                       }

                            df = pd.DataFrame([results], columns=results_header)
                            append_df_to_csv(df, all_results_filename, sep=',', header=results_header)

                            if best_in_class == [] or best_in_class['rmse'] > results['rmse']:
                                best_in_class = results

    append_df_to_csv(pd.DataFrame([best_in_class], columns=results_header), best_results_filename, sep=','
                     , header=results_header)

print('All finished')
