import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
import numpy as np
import os
from sklearn.linear_model import SGDRegressor

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "petrol_consumption.csv"
dataset = pd.read_csv(PATH + CSV_DATA)
#   Petrol_Consumption
X = dataset[['Petrol_tax', 'Average_income', 'Population_Driver_licence(%)']]

x_with_const = sm.add_constant(X)
y = dataset['Petrol_Consumption'].values

X_train, X_test, y_train, y_test = train_test_split(x_with_const, y,
                                                    test_size=0.2, random_state=0)


def perform_linear_regression(X_train, X_test, y_train, y_test):
    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model
    print(model.summary())
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    return predictions


predictions = perform_linear_regression(X_train, X_test, y_train, y_test)

def perform_sgd(X_train, X_test, y_train, y_test):
    sgd = SGDRegressor()
    sgd.fit(X_train, y_train)
    print("\n***SGD=")
    predictions = sgd.predict(X_test)

    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test,
                                             predictions)))


perform_sgd(X_train, X_test,
            y_train, y_test)
