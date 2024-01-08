import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "USA_Housing.csv"
dataset = pd.read_csv(PATH + CSV_DATA)

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)
print(dataset.head())
print(dataset.describe())

X = dataset[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
             "Area Population", 'Price']]

# Adding an intercept (required)
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X = sm.add_constant(X)
y = dataset[['Price']]

k_fold = KFold(n_splits=5, shuffle=True)

rmses = []
for train_index, test_index in k_fold.split(X):
    # use index lists to isolate rows for train and test sets.
    # Get rows filtered by index and all columns.
    # X.loc[row number array, all columns]
    X_train = X.loc[X.index.intersection(train_index), :]
    X_test = X.loc[X.index.intersection(test_index), :]
    y_train = y.loc[y.index.intersection(train_index), :]
    y_test = y.loc[y.index.intersection(test_index), :]

    # build model
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(predictions, y_test)
    rmse = np.sqrt(mse)
    print("RMSE: " + str(rmse))
    rmses.append(rmse)

avg_rmse = np.mean(rmses)
print("Average rmse: " + str(avg_rmse))
