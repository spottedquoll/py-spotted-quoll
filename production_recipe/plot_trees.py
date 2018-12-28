import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from production_recipe.helpers import get_recipe_df, perform_cleaning
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt

recipe_directory = '/Volumes/slim/emily1_projects/2017_ProductionRecipes/'
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
df3 = df2.drop('a', axis=1)

# Convert to arrays
y = np.array(df2['a'])
X = np.array(df3)

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

# Export as dot file
export_graphviz(regressor, out_file=results_dir+'tree.dot', feature_names=df3.columns, class_names=['a'],
                rounded=True, proportion=False, precision=2, filled=True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', results_dir + 'tree.dot', '-o', results_dir + 'tree.png', '-Gdpi=600'])

plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('tree.png'))
plt.axis('off');
plt.show();

# export_graphviz(regressor, out_file=results_dir+'tree.dot', feature_names=df3.columns, class_names=['a'],
#                 rounded=True, proportion=False, precision=2, filled=True, max_depth=5)