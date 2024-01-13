import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
import numpy as np
import os
from sklearn.linear_model import Lasso

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "petrol_consumption.csv"
dataset = pd.read_csv(PATH + CSV_DATA)
#   Petrol_Consumption
X = dataset[['Petrol_tax', 'Average_income', 'Population_Driver_licence(%)']]

x_with_const = sm.add_constant(X)
y = dataset['Petrol_Consumption'].values

X_train, X_test, y_train, y_test = train_test_split(x_with_const, y,
                                                    test_size=0.2, random_state=42)


def perform_linear_regression(X_train, X_test, y_train, y_test):
    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model
    print(model.summary())
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    return predictions


predictions = perform_linear_regression(X_train, X_test, y_train, y_test)

def perform_lasso_regression(X_train, X_test, y_train, y_test, alpha):
    lassoreg = Lasso(alpha=alpha)
    lassoreg.fit(X_train, y_train)
    y_pred = lassoreg.predict(X_test)
    print("\n***Lasso Regression Coefficients ** alpha=" + str(alpha))
    print(lassoreg.intercept_)
    print(lassoreg.coef_)
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
alpha_values = [0, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
for i in range(0, len(alpha_values)):
    perform_lasso_regression(X_train, X_test, y_train,
                             y_test, alpha_values[i])
