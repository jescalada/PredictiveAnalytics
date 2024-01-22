import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
import os

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load and prepare data.
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
FOLDER = f"{ROOT_PATH}\\..\\datasets\\"
FILE = 'petrol_consumption.csv'
dataset = pd.read_csv(FOLDER + FILE)
print(dataset)
X = dataset.copy()
del X['Petrol_Consumption']
y = dataset[['Petrol_Consumption']]

feature_combo_list = []


def evaluate_model(model, X_test, y_test, title, num_estimators, max_features, max_samples):
    print("\n****** " + title)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Store statistics and add to list.
    stats = {"type": title, "rmse": rmse,
             "estimators": num_estimators, "features": max_features, "samples": max_samples}
    feature_combo_list.append(stats)


num_estimator_list = [750, 800, 900, 1000]
max_features_list = [3, 4]
max_samples_list = [0.2, 0.4, 0.6]

for num_estimators in num_estimator_list:
    for max_features in max_features_list:
        for max_samples in max_samples_list:
            # Create random split.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


            # Build linear regression ensemble.
            ensemble_model = BaggingRegressor(estimator=LinearRegression(),
                                              max_features=max_features,
                                              # Can be percent (float) or actual
                                              # total of samples (int).
                                              max_samples=max_samples,
                                              n_estimators=num_estimators).fit(X_train, y_train.values.ravel())
            evaluate_model(ensemble_model, X_test, y_test, "Ensemble",
                           num_estimators, max_features, max_samples)

            # Build stand alone linear regression model.
            model = LinearRegression()
            model.fit(X_train, y_train)
            evaluate_model(model, X_test, y_test, "Linear Regression", None, None, None)

# Build data frame with dictionary objects.
df_stats = pd.DataFrame()
print(df_stats)
for combo in feature_combo_list:
    df_stats = pd.concat([df_stats,
                          pd.DataFrame.from_records([combo])],
                         ignore_index=True)

# Sort and show all combinations.
# Show all rows
pd.set_option('display.max_rows', None)
df_stats = df_stats.sort_values(by=['type', 'rmse'])
print(df_stats)
