import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from production_recipe.helpers import get_recipe_df, perform_cleaning

recipe_directory = '/Volumes/slim/2017_ProductionRecipes/'
dataset_dir = recipe_directory + '/data/'
results_dir = recipe_directory + '/results/'

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

training_cols = list(df2.columns)
del training_cols[0]

# Post scaling, region y to X, to enable sampling
train_scaled_df = pd.concat([pd.DataFrame(y_train, columns=['a']), pd.DataFrame(X_train_scale, columns=training_cols)]
                            , axis=1)

# Fold settings
folds = 7
sample_frac = 1/5
sub_models = []

for i in range(folds):

    print('Folding ' + str(i))

    # Sample from whole data set
    sampled_df = train_scaled_df.sample(frac=sample_frac, replace=False)
    y_train_sample = np.array(sampled_df['a'])
    X_train_sample = np.array(sampled_df.drop('a', axis=1))

    # Predict
    regressor = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto', metric='cosine')
    regressor.fit(X_train_sample, y_train_sample)
    y_pred = regressor.predict(X_test_scale)

    # Store model meta
    meta = {'fold': i, 'model': regressor, 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))}
    sub_models.append(meta)

    print('.')

# Calculate ensemble prediction
prediction_array = []
weights = []
for estimator in sub_models:
    regressor = estimator['model']
    prediction_array.append(regressor.predict(X_test_scale))
    weights.append(1/estimator['rmse'])

normal_weights = weights / sum(weights)
y_pred = np.average(prediction_array, axis=0, weights=normal_weights)

print('mae: ' + str(mean_absolute_error(y_test, y_pred))
      + ', rmse: ' + str(np.sqrt(mean_squared_error(y_test, y_pred)))
      + ', r2: ' + str(r2_score(y_test, y_pred))
      )