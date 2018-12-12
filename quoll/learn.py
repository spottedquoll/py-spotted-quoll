import numpy as np
from sklearn.metrics import mean_squared_error


def ensemble_predict(sub_models, x_test):
    prediction_array = []
    weights = []
    for estimator in sub_models:
        regressor = estimator['model']
        y_pred = regressor.predict(x_test)
        prediction_array.append(y_pred)
        weights.append(1 / estimator['rmse'])

    normal_weights = weights / sum(weights)
    y_pred = np.average(prediction_array, axis=0, weights=normal_weights)

    return y_pred


def bag_ensemble(regressor, training_df, x_test, y_test, y_col_name='y', folds=5, sample_fraction=10):

    sub_models = []

    for i in range(folds):

        # Sample from whole data set
        sampled_df = training_df.sample(frac=sample_fraction, replace=False)
        y_train_sample = np.array(sampled_df[y_col_name])
        x_train_sample = np.array(sampled_df.drop(y_col_name, axis=1))

        # Predict
        regressor.fit(x_train_sample, y_train_sample)
        y_pred = regressor.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Store model meta
        meta = {'fold': i, 'model': regressor, 'rmse': rmse}
        sub_models.append(meta)

    return sub_models