from sklearn import metrics
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "winequality.csv"
dataset = pd.read_csv(PATH + CSV_DATA)

X = dataset[['volatile acidity', 'chlorides', 'total sulfur dioxide', 'sulphates',
             'alcohol']]

X_withConst = sm.add_constant(X)
y = dataset['quality'].values


# Show all columns.
pd.set_option('display.max_columns', None)

# Include only statistically significant columns.
X = dataset[['volatile acidity', 'chlorides', 'total sulfur dioxide',
             'pH', 'sulphates', 'alcohol']]
X = sm.add_constant(X)
y = dataset['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Stochastic gradient descent models are sensitive to scaling.
# Fit X scaler and transform X_train.
from sklearn.preprocessing import StandardScaler

scalerX = StandardScaler()
X_train_scaled = scalerX.fit_transform(X_train)

# Build y scaler and transform y_train.
scalerY = StandardScaler()
y_train_scaled = scalerY.fit_transform(np.array(y_train).reshape(-1, 1))

# Scale test data.
X_test_scaled = scalerX.transform(X_test)


def perform_linear_regression(X_train, X_test, y_train, y_test, scalerY):
    model = sm.OLS(y_train, X_train).fit()
    scaled_predictions = model.predict(X_test)  # make the predictions by the model
    predictions = scalerY.inverse_transform(np.array(scaled_predictions).reshape(-1, 1))
    print(model.summary())
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    return predictions


predictions = perform_linear_regression(X_train_scaled, X_test_scaled,
                                        y_train_scaled, y_test, scalerY)

from sklearn.linear_model import SGDRegressor

def performSGD(X_train, X_test, y_train, y_test, scalerY):
    sgd = SGDRegressor()
    sgd.fit(X_train, y_train)
    print("\n***SGD=")
    predictionsUnscaled = sgd.predict(X_test)
    predictions = scalerY.inverse_transform(
        np.array(predictionsUnscaled).reshape(-1,1))

    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test,
                                             predictions)))


performSGD(X_train_scaled, X_test_scaled,
           y_train_scaled, y_test, scalerY)

