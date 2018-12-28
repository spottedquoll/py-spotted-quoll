import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.neighbors import KNeighborsRegressor
from production_recipe.helpers import get_recipe_df, perform_cleaning
from quoll.utils import append_df_to_csv

# initialise paths
recipe_directory = '/Volumes/slim/2017_ProductionRecipes/'
dataset_dir = recipe_directory + '/data/'
results_dir = recipe_directory + '/results/'

all_results_filename = results_dir + 'model_performance_summary.csv'
results_header = ['model', 'hyper-parameters', 'mae', 'mse', 'rmse', 'r2', 'explained_variance', 'timestamp']

# get raw data
df, header, year_labels = get_recipe_df(dataset_dir, 25)
df = perform_cleaning(df)

# One hot encode categorical variables
df2 = pd.get_dummies(df, columns=['country', 'margin'])  # , 'i', 'j'

# Convert to arrays
y = np.array(df2['a'])
X = np.array(df2.drop('a', axis=1))

# Split training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scale = 'discrete'

# Scale the continuous data
if scale is 'minmax':
    scaler = MinMaxScaler()
    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)
elif scale is 'discrete':
    scaler = KBinsDiscretizer(n_bins=10, encode='onehot-dense')
    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)
elif scale is None:
    X_train_scale = X_train
    X_test_scale = X_test

# Column names
training_cols = list(df2.columns)
del training_cols[0]

# Post scaling, region y to X, to enable sampling
train_scaled_df = pd.concat([pd.DataFrame(y_train, columns=['a']), pd.DataFrame(X_train_scale)], axis=1)  # , columns=training_cols)

model_type = 'k_nearest_neighbors_bagged'

# Fold settings
n = 5
f = 15
fraction = 0.1
m = 'cosine'
w = 'distance'
a = 'auto'
sub_models = []

for i in range(f):

    print('Folding ' + str(i))

    # Sample from whole data set
    sampled_df = train_scaled_df.sample(frac=fraction, replace=False)
    y_train_sample = np.array(sampled_df['a'])
    X_train_sample = np.array(sampled_df.drop('a', axis=1))

    # Predict
    regressor = KNeighborsRegressor(n_neighbors=n, weights=w, algorithm=a, metric=m)
    regressor.fit(X_train_sample, y_train_sample)

    y_pred = regressor.predict(X_test_scale)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Store model meta
    meta = {'fold': i, 'model': regressor, 'rmse': rmse}
    sub_models.append(meta)

    print('.')

print('Calculating ensemble prediction...')

prediction_array = []
weights = []
i = 1

for estimator in sub_models:

    print(str(i) + ' of ' + str(f))

    regressor = estimator['model']

    y_pred = regressor.predict(X_test_scale)
    prediction_array.append(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    weights.append(1/estimator['rmse'])

    print('.')
    i = i + 1

normal_weights = weights / sum(weights)
y_pred = np.average(prediction_array, axis=0, weights=normal_weights)

print('mae: ' + str(mean_absolute_error(y_test, y_pred))
      + ', rmse: ' + str(np.sqrt(mean_squared_error(y_test, y_pred)))
      + ', r2: ' + str(r2_score(y_test, y_pred))
      )

# Save results
results = {'model': model_type, 'hyper-parameters': [n, w, a, f, fraction, m, scale]
           , 'mae': mean_absolute_error(y_test, y_pred)
           , 'mse': mean_squared_error(y_test, y_pred)
           , 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
           , 'r2': r2_score(y_test, y_pred)
           , 'explained_variance': explained_variance_score(y_test, y_pred)
           , 'timestamp': str(datetime.datetime.now())
           }

df = pd.DataFrame([results], columns=results_header)
append_df_to_csv(df, all_results_filename, sep=',', header=results_header)