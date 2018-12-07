import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_ridge import KernelRidge
from production_recipe.helpers import get_recipe_df, perform_cleaning
from quoll.utils import mixed_list_to_string

recipe_directory = '/Volumes/slim/2017_ProductionRecipes/'
dataset_dir = recipe_directory + '/data/'
results_dir = recipe_directory + '/results/'

df, header, year_labels = get_recipe_df(dataset_dir, 25)

df = perform_cleaning(df)

# One hot encode categorical variables
df2 = pd.get_dummies(df, columns=['country', 'margin'])  # , 'i', 'j'

kernel_opts = ['linear', 'poly', 'rbf']
gamma_opts = [None, 0.1, 1, 10]
alpha_opts = [1e-4, 1e-3, 1e-2, 1, 5, 10]
scaling_opts = [True, False]
folds = 10

model_type = 'kernel_ridge_regression'
print('Building ' + model_type + ' models')

for i in range(5):

    # Reduce sample size
    df3 = df2.sample(frac=0.1, replace=False)
    print('Sampled ' + str(len(df3)) + ' records.')

    # Convert to arrays
    y = np.array(df3['a'])
    X = np.array(df3.drop('a', axis=1))

    # Split training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Scale the continuous data
    scaler = MinMaxScaler()
    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)

    results = []

    X_train_temp = X_train_scale
    X_test_temp = X_test_scale

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

                    print('mae: ' + str(metrics.mean_absolute_error(y_test, y_pred))
                          + ', rmse: ' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                          + ', r2: ' + str(metrics.r2_score(y_test, y_pred))
                          + ', opts: ' + mixed_list_to_string([i, k, g, a, s], ',')
                          )