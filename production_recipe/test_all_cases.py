import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from production_recipe.helpers import get_recipe_df, perform_cleaning
from quoll.utils import append_df_to_csv

# Set paths
recipe_directory = '/Volumes/slim/more_projects/emily1/Projects/2017_ProductionRecipes/'
dataset_dir = recipe_directory + '/data/'
results_dir = recipe_directory + '/results/'

# Create results file
results_filename = results_dir + 'prediction_summary.csv'
results_header = ['model', 'agg', 'trial', 'mae', 'mse', 'rmse', 'r2', 'explained_variance', 'timestamp']

# options
trials = 5
aggregations = [25, 100]

for agg in aggregations:

    # read data and clean
    df, header, year_labels = get_recipe_df(dataset_dir, agg)
    df = perform_cleaning(df)

    # One hot encode categorical variables
    df2 = pd.get_dummies(df, columns=['country', 'margin'])

    # Convert to arrays
    y = np.array(df2['a'])
    X = np.array(df2.drop('a', axis=1))

    for run in np.arange(trials):

        # Split training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        # Build model
        regressor = RandomForestRegressor(max_depth=None, min_samples_leaf=1, max_features='auto'
                                          , n_estimators=100, min_samples_split=5)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

        results = {'model': 'random_forest', 'agg': agg, 'trial': run
                   , 'mae': mean_absolute_error(y_test, y_pred)
                   , 'mse': mean_squared_error(y_test, y_pred)
                   , 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                   , 'r2': r2_score(y_test, y_pred)
                   , 'explained_variance': explained_variance_score(y_test, y_pred)
                   , 'timestamp': str(datetime.datetime.now())
                   }

        assert(len(results) == len(results_header))

        append_df_to_csv(pd.DataFrame([results], columns=results_header), results_filename, sep=','
                         , header=results_header)

    print('All finished')
