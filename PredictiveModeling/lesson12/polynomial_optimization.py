import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os
import numpy as np

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = ROOT_PATH + "\\..\\datasets\\"
CSV_DATA = "usCityData.csv"
df = pd.read_csv(PATH + CSV_DATA)
X = df[['lstat']]
y = df[['medv']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, \
                                                    random_state=42, shuffle=True)

# Polynomial Regression-nth order
plt.scatter(X_test, y_test, s=10, alpha=0.3)

for degree in [1, 2, 3, 4, 5, 6, 7]:
    # This line of code generates
    # ['1', 'x0', 'x0^2', 'x0^3', 'x0^4']
    # for all x values.
    poly_reg = PolynomialFeatures(degree=degree)
    transformed_x = poly_reg.fit_transform(X_train)

    # Generate intercept and coefficient.
    linReg = LinearRegression()
    linReg.fit(transformed_x, y_train)

    # Generate predictions and evaluate R^2.
    transformed_x = poly_reg.fit_transform(X_test)
    predictions = linReg.predict(transformed_x)

    rmse = np.sqrt(np.sum((y_test - predictions) ** 2) / len(predictions))

    # Plot the predictions along with the RMSE
    plt.plot(X_test, predictions, label="degree %d" % degree + \
                ", RMSE %.2f" % rmse)

# Show the plots for all polynomial equations.
plt.legend(loc='upper right')
plt.xlabel("LSTAT ")
plt.ylabel("MEDV")
plt.title("Fit for Polynomial Models")
plt.show()

