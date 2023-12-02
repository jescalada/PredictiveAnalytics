import pandas as pd
from sklearn.linear_model import LinearRegression
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = ROOT_PATH + "\\..\\datasets\\"
CSV_DATA = "salaries.csv"

# Note this has a comma separator.
df       = pd.read_csv(PATH + CSV_DATA)

# Extract target and salary.
X        = df[['Level']]
y        = df[['Salary']]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                            test_size=0.2, random_state=0)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)

# This line of code generates
# ['1', 'x0', 'x0^2', 'x0^3', 'x0^4']
# for all x values.
X_poly   = poly_reg.fit_transform(X_train)

# Generate intercept and coefficient.
linReg   = LinearRegression()
linReg.fit(X_poly, y_train)

print("\ncoefficients: ")
print(linReg.coef_)

print("\nIntercept: ")
print(linReg.intercept_)

poly_reg.fit_transform([[3]])

# Generate predictions.
# Predicting a new result with Polymonial Regression
predictions = linReg.predict(poly_reg.fit_transform(X_test))
print("\nPredictions")
print(predictions)

import numpy as np
RMSE = np.sqrt(np.sum((y_test - predictions)**2)/len(predictions))
print("RMSE: " + str(RMSE))
