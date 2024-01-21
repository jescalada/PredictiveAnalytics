import pandas as pd
from sklearn.metrics import mean_squared_error
import os
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE = 'temperatures.csv'

# Read in data and display first 5 rows
features = pd.read_csv(PATH + FILE)

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(features)

# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)

# Display the first 5 rows of the last 12 columns.
print(features.head(5))

# Labels are the values we want to predict
labels = np.array(features['actual'])

# Remove the labels from the features
# axis 1 refers to the columns
features = features.drop('actual', axis=1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)


# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = \
    train_test_split(features, labels, test_size=0.25, random_state=42)

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=400, min_samples_split=15, min_samples_leaf=15, max_features=0.5, max_depth=None, bootstrap=True, random_state=42)

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')

# Print out the mean square error.
mse = mean_squared_error(test_labels, predictions)
print('RMSE:', np.sqrt(mse))

# Get numerical feature importances
importances = list(rf.feature_importances_)


# Present features and importance scores.
def show_feature_importances(importances, feature_list):
    df_importance = pd.DataFrame()
    for i in range(0, len(importances)):
        df_importance = df_importance._append({"importance": importances[i],
                                             "feature": feature_list[i]},
                                            ignore_index=True)

    df_importance = df_importance.sort_values(by=['importance'],
                                            ascending=False)
    print(df_importance)


show_feature_importances(importances, feature_list)


random_grid = \
    {'bootstrap': [True],
     'max_depth': [4, 6, None],
     'max_features': [0.5],
     'min_samples_leaf': [15],
     'min_samples_split': [15],
     'n_estimators': [400, 800, 1600]}

print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, n_jobs=-1)
# Fit the random search model
rf_random.fit(train_features, train_labels)

print("Best parameters")
print(rf_random.best_params_)
