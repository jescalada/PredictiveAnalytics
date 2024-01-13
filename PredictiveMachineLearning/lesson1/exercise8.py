import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
import numpy as np
import os
from sklearn.linear_model import Ridge

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "petrol_consumption.csv"
dataset = pd.read_csv(PATH + CSV_DATA)
#   Petrol_Consumption
X = dataset[['Petrol_tax', 'Average_income', 'Population_Driver_licence(%)']]

x_with_const = sm.add_constant(X)
y = dataset['Petrol_Consumption'].values

X_train, X_test, y_train, y_test = train_test_split(x_with_const, y,
                                                    test_size=0.2)


def perform_linear_regression(X_train, X_test, y_train, y_test):
    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model
    print(model.summary())
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    return predictions


predictions = perform_linear_regression(X_train, X_test, y_train, y_test)

def ridge_regression(X_train, X_test, y_train, y_test, alpha):
    # Fit the model
    ridgereg = Ridge(alpha=alpha)
    ridgereg.fit(X_train, y_train)
    y_pred = ridgereg.predict(X_test)

   # predictions = scalerY.inverse_transform(y_pred.reshape(-1,1))
    print("\n***Ridge Regression Coefficients ** alpha=" + str(alpha))
    print(ridgereg.intercept_)
    print(ridgereg.coef_)
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                         test_size=0.2)
alphaValues = [0,  0.02, 0.04, 0.06, 0.08, 0.1]
for i in range(0, len(alphaValues)):
    ridge_regression(X_train, X_test, y_train,
                     y_test, alphaValues[i])
