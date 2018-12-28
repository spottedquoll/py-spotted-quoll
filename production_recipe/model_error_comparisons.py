import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from production_recipe.helpers import get_recipe_df, perform_cleaning
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from quoll.learn import bag_ensemble, ensemble_predict

recipe_directory = '/Volumes/slim/2017_ProductionRecipes/'
dataset_dir = recipe_directory + '/data/'
results_dir = recipe_directory + '/results/'

df, header, year_labels = get_recipe_df(dataset_dir, 25)

df = perform_cleaning(df)

predictions = []
errors = []
absolute_errors = []
series_labels = []

# One hot encode categorical variables
df2 = pd.get_dummies(df, columns=['country', 'margin'])

# Convert to arrays
y = np.array(df2['a'])
X = np.array(df2.drop('a', axis=1))

# Split training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Scale the continuous data
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

# Build models

# Decision tree
model_type = 'Decision tree'
print('Building ' + model_type + ' model')
regressor = DecisionTreeRegressor(max_depth=None, min_samples_leaf=1, max_features='auto'
                                  , splitter='best', criterion='mse', min_samples_split=5)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

errors.append(y_test - y_pred)
absolute_errors.append(abs(y_test - y_pred))
series_labels.append(model_type)
predictions.append(y_pred)

# random forest
model_type = 'Random forest'
print('Building ' + model_type + ' model')

regressor = RandomForestRegressor(max_depth=None, min_samples_leaf=1, max_features='auto'
                                  , n_estimators=100, min_samples_split=2)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

errors.append(y_test - y_pred)
absolute_errors.append(abs(y_test - y_pred))
series_labels.append(model_type)
predictions.append(y_pred)

# ada boost
model_type = 'Ada boost'
print('Building ' + model_type + ' model')

regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=None, splitter='best'
                                                    , criterion='mse', min_samples_leaf=10)
                              , n_estimators=150, loss='exponential', learning_rate=0.01)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

errors.append(y_test - y_pred)
absolute_errors.append(abs(y_test - y_pred))
series_labels.append(model_type)
predictions.append(y_pred)

# gradient boost
model_type = 'Gradient boost'
print('Building ' + model_type + ' model')

regressor = GradientBoostingRegressor(n_estimators=250, loss='ls', learning_rate=1, max_features='auto',
                                      min_samples_leaf=5, min_samples_split=2)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

errors.append(y_test - y_pred)
absolute_errors.append(abs(y_test - y_pred))
series_labels.append(model_type)
predictions.append(y_pred)

# MPL
model_type = 'MPL'
print('Building ' + model_type + ' model')

regressor = MLPRegressor(alpha=1e-5, solver='adam', activation='relu'
                         , hidden_layer_sizes=(round(df2.shape[1]*1.5),) * 5, learning_rate='adaptive')

regressor.fit(X_train_scale, y_train)
y_pred = regressor.predict(X_test_scale)

errors.append(y_test - y_pred)
absolute_errors.append(abs(y_test - y_pred))
series_labels.append(model_type)
predictions.append(y_pred)

# KNN
model_type = 'KNN'
print('Building ' + model_type + ' model')

# Post scaling, rejoin y to X to enable sampling
training_cols = list(df2.columns)
del training_cols[0]

train_scaled_df = pd.concat(
    [pd.DataFrame(y_train, columns=['a']), pd.DataFrame(X_train_scale, columns=training_cols)]
    , axis=1)
X_test_temp = X_test_scale

regressor = KNeighborsRegressor(n_neighbors=3, weights='distance', algorithm='auto', metric='cosine')

sub_models = bag_ensemble(regressor, train_scaled_df, X_test_temp, y_test
                          , y_col_name='a', folds=20, sample_fraction=0.2)

y_pred = ensemble_predict(sub_models, X_test_temp)

errors.append(y_test - y_pred)
absolute_errors.append(abs(y_test - y_pred))
series_labels.append(model_type)
predictions.append(y_pred)

# filter
abs_errors_filt = []
for k in absolute_errors:
    abs_errors_filt.append(k[(k > 10e-6)])

errors_filt = []
for k in errors:
    errors_filt.append(k[(k > 10e-6) | (k < -10e-6)])

# Box plots
flier_props = dict(markersize=2, markerfacecolor='grey', marker='.', alpha=0.4, markeredgecolor='grey')
fig = plt.figure()
plt.boxplot(abs_errors_filt, labels=series_labels, vert=True, flierprops=flier_props)
plt.yscale('log')
plt.ylabel('Absolute error')
plt.xticks(rotation=45, fontsize=10)
plt.savefig(results_dir + 'prediction_error_boxplots.png', dpi=700, bbox_inches='tight')
plt.clf()

# histogram (combined)
bins = np.linspace(-0.5, 0.5, 60)
fig = plt.figure()
for e in errors:
    plt.hist(e, bins, density=True, color='cornflowerblue', ec='cornflowerblue', alpha=0.6)  # , histtype='stepfilled'
plt.grid(axis='y', alpha=0.75)
plt.legend(series_labels, loc='upper center', bbox_to_anchor=(1.2, 0.8), fontsize='small', frameon=False)
plt.savefig(results_dir + 'prediction_error_histogram.png', dpi=700, bbox_inches='tight')
plt.clf()

# histogram (subplots)
bins = np.linspace(-0.25, 0.25, 60)
f, a = plt.subplots(2, 3)
a = a.ravel()
rows = [0, 2]
for idx, ax in enumerate(a):
    bin_weights = np.ones_like(errors[idx]) / float(len(errors[idx]))
    ax.hist(errors[idx], bins=bins, weights=bin_weights)
    ax.set_title(series_labels[idx])
    ax.set_ylim(0, 0.6)
    #ax.set_xlabel(xaxes[idx])
    if idx in rows:
        ax.set_ylabel('Absolute error')
plt.tight_layout(pad=0.75, w_pad=1.0, h_pad=1.0)
plt.savefig(results_dir + 'prediction_error_histogram_subplots.png', dpi=700, bbox_inches='tight')
plt.clf()

# scatter
fig = plt.figure()
for idx, y_pred in enumerate(predictions):

    plt.subplot(2, 3, idx+1)
    plt.scatter(y_test, y_pred, s=5, c='black')
    plt.ylim(0, 1)
    plt.xlim(0, 1)

    plt.xlabel('Aij (actual)', fontsize=8, labelpad=8)
    plt.ylabel('Aij (predicted)', fontsize=8, labelpad=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.title(series_labels[idx], fontsize=8)

plt.tight_layout(pad=0.75, w_pad=1.5, h_pad=0.3)
plt.savefig(results_dir + 'prediction_correlation_scatter.png', dpi=700, bbox_inches='tight')
plt.clf()

# Make table
table = []
for idx, y_pred in enumerate(predictions):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results = {'model': series_labels[idx], 'mae': mean_absolute_error(y_test, y_pred)
               , 'mse': mean_squared_error(y_test, y_pred)
               , 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
               , 'r2': r2_score(y_test, y_pred)
               }
    table.append(results)

header = ['model', 'mae', 'rmse', 'r2']
df = pd.DataFrame(table, columns=header)
df.to_csv(results_dir + 'best_models_summary_table.csv', sep=',', header=header)