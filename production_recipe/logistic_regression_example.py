import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from quoll.utils import mixed_list_to_string
from production_recipe.helpers import get_recipe_df, perform_cleaning


recipe_directory = '/Volumes/slim/2017_ProductionRecipes/'
dataset_dir = recipe_directory + '/data/'

df, header, year_labels = get_recipe_df(dataset_dir, 25)
df = perform_cleaning(df)

# One hot encode categorical variables
df2 = pd.get_dummies(df, columns=['i', 'j', 'country', 'margin',])

# Convert to arrays
y = np.array(df2['a'])
X = np.array(df2.drop('a', axis=1))

# Split training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

model_type = 'logistic_regression'
penalties = ['l1', 'l2']
gamma_opts = [0.1, 1, 10]
intercept_opts = [True, False]

print('Building ' + model_type + ' models')
best_in_class = []

for p in penalties:
    for g in gamma_opts:
        for i in intercept_opts:

            regressor = LogisticRegression(C=g, class_weight=None, dual=False, fit_intercept=i
                                           , intercept_scaling=1, max_iter=20, multi_class='ovr', n_jobs=3
                                           , penalty=p, solver='liblinear', tol=0.0001)

            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)

            print('mae: ' + str(mean_absolute_error(y_test, y_pred))
                  + ', rmse: ' + str(np.sqrt(mean_squared_error(y_test, y_pred)))
                  + ', r2: ' + str(r2_score(y_test, y_pred))
                  + ', opts: ' + mixed_list_to_string([p, g, i], ',')
                 )