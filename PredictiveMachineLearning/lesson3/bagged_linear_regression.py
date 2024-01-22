import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
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

# Create random split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def evaluate_model(model, X_test, y_test, title):
    print("\n****** " + title)
    predictions = model.predict(X_test)
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, predictions)))


# Build linear regression ensemble.
ensemble_model = BaggingRegressor(base_estimator=LinearRegression(), max_features=4,
                                  max_samples=0.5,
                                  n_estimators=10).fit(X_train, y_train)
evaluate_model(ensemble_model, X_test, y_test, "Ensemble")

# Build stand alone linear regression model.
model = LinearRegression()
model.fit(X_train, y_train)
evaluate_model(model, X_test, y_test, "Linear Regression")
