import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, Ridge, Lasso

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

def perform_SGD(X_train, X_test, y_train, y_test, scalerY):
    sgd = SGDRegressor()
    sgd.fit(X_train, y_train)
    print("\n***SGD=")
    predictions_unscaled = sgd.predict(X_test)
    predictions = scalerY.inverse_transform(
        np.array(predictions_unscaled).reshape(-1,1))

    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test,
                                             predictions)))


perform_SGD(X_train_scaled, X_test_scaled,
            y_train_scaled, y_test, scalerY)

def ridge_regression(X_train, X_test, y_train, y_test, alpha, scalerY):
    # Fit the model
    ridgereg = Ridge(alpha=alpha)
    ridgereg.fit(X_train, y_train)
    y_pred_scaled = ridgereg.predict(X_test)

   # predictions = scalerY.inverse_transform(y_pred.reshape(-1,1))
    print("\n***Ridge Regression Coefficients ** alpha=" + str(alpha))
    print(ridgereg.intercept_)
    print(ridgereg.coef_)
    y_pred = scalerY.inverse_transform(np.array(y_pred_scaled).reshape(-1,1))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                         test_size=0.2, random_state=0)
alphaValues = [0,  0.16, 0.17, 0.18]
for i in range(0, len(alphaValues)):
    ridge_regression(X_train_scaled, X_test_scaled, y_train_scaled,
                     y_test, alphaValues[i], scalerY)


def perform_lasso_regression(X_train, X_test, y_train, y_test, alpha,
                             scalerY):
    lassoreg = Lasso(alpha=alpha)
    lassoreg.fit(X_train, y_train)
    y_pred_scaled = lassoreg.predict(X_test)
    y_pred = scalerY.inverse_transform(np.array(y_pred_scaled).reshape(-1,1))
    print("\n***Lasso Regression Coefficients ** alpha=" + str(alpha))
    print(lassoreg.intercept_)
    print(lassoreg.coef_)
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
alpha_values = [0, 0.1, 0.5, 1]
for i in range(0, len(alpha_values)):
    perform_lasso_regression(X_train_scaled, X_test_scaled, y_train_scaled,
                             y_test, alpha_values[i], scalerY)
