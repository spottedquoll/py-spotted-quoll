import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from production_recipe.helpers import get_recipe_df, perform_cleaning
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

recipe_directory = '/Volumes/slim/2017_ProductionRecipes/'
dataset_dir = recipe_directory + '/data/'
results_dir = recipe_directory + '/results/'

df, header, year_labels = get_recipe_df(dataset_dir, 25)

df = perform_cleaning(df)

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
model_type = 'decision_tree'
print('Building ' + model_type + ' model')
regressor = DecisionTreeRegressor(max_depth=None, min_samples_leaf=1, max_features='auto'
                                  , splitter='best', criterion='mse', min_samples_split=5)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

errors.append(y_test - y_pred)
absolute_errors.append(abs(y_test - y_pred))
series_labels.append(model_type)

# random forest
model_type = 'random_forest'
print('Building ' + model_type + ' model')

regressor = RandomForestRegressor(max_depth=None, min_samples_leaf=1, max_features='auto'
                                  , n_estimators=100, min_samples_split=2)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

errors.append(y_test - y_pred)
absolute_errors.append(abs(y_test - y_pred))
series_labels.append(model_type)

print('model: ' + model_type + ', mae: ' + str(mean_absolute_error(y_test, y_pred))
      + ', rmse: ' + str(np.sqrt(mean_squared_error(y_test, y_pred)))
      + ', r2: ' + str(r2_score(y_test, y_pred))
      )

# ada boost
model_type = 'ada_boost'
print('Building ' + model_type + ' model')

regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=None, splitter='best'
                                                    , criterion='mse', min_samples_leaf=10)
                              , n_estimators=150, loss='exponential', learning_rate=0.01)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

errors.append(y_test - y_pred)
absolute_errors.append(abs(y_test - y_pred))
series_labels.append(model_type)

# gradient boost
model_type = 'gradient_boost'
print('Building ' + model_type + ' model')

regressor = GradientBoostingRegressor(n_estimators=200, loss='ls', learning_rate=1, max_features='auto')

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

errors.append(y_test - y_pred)
absolute_errors.append(abs(y_test - y_pred))
series_labels.append(model_type)

# MPL
model_type = 'mpl'
print('Building ' + model_type + ' model')

regressor = MLPRegressor(alpha=0.001, solver='adam', activation='relu')

regressor.fit(X_train_scale, y_train)
y_pred = regressor.predict(X_test_scale)

errors.append(y_test - y_pred)
absolute_errors.append(abs(y_test - y_pred))
series_labels.append(model_type)

# Ridge
model_type = 'ridge'
print('Building ' + model_type + ' model')

regressor = Ridge(alpha=20, normalize=False, fit_intercept=True)

regressor.fit(X_train_scale, y_train)
y_pred = regressor.predict(X_test_scale)

errors.append(y_test - y_pred)
absolute_errors.append(abs(y_test - y_pred))
series_labels.append(model_type)

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
for idx, val in enumerate(ints):

    plt.subplot(2, 3, idx)
    plt.savefig(results_dir + 'prediction_error_boxplots.png', dpi=700, bbox_inches='tight')
    plt.clf()