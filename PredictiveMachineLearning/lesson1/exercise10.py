from sklearn.linear_model import ElasticNet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
import numpy as np
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "petrol_consumption.csv"
dataset = pd.read_csv(PATH + CSV_DATA)
#   Petrol_Consumption
X = dataset[['Petrol_tax', 'Average_income', 'Population_Driver_licence(%)']]

x_with_const = sm.add_constant(X)
y = dataset['Petrol_Consumption'].values

X_train, X_test, y_train, y_test = train_test_split(x_with_const, y,
                                                    test_size=0.2, random_state=3)


def perform_linear_regression(X_train, X_test, y_train, y_test):
    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model
    print(model.summary())
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    return predictions


predictions = perform_linear_regression(X_train, X_test, y_train, y_test)

best_rmse = 100000.03


def perform_elastic_net_regression(X_train, X_test, y_train, y_test, alpha, l1ratio, best_rmse,
                                   best_alpha, best_l1_ratio):
    model = ElasticNet(alpha=alpha, l1_ratio=l1ratio)
    # fit model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n***ElasticNet Regression Coefficients ** alpha=" + str(alpha)
          + " l1ratio=" + str(l1ratio))
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print(model.intercept_)
    print(model.coef_)
    try:
        if (rmse < best_rmse):
            best_rmse = rmse
            best_alpha = alpha
            best_l1_ratio = l1ratio
        print('Root Mean Squared Error:', rmse)
    except:
        print("rmse =" + str(rmse))

    return best_rmse, best_alpha, best_l1_ratio


alpha_values = [0, 0.00001, 0.0001, 0.001, 0.01, 0.18]
l1ratioValues = [0, 0.25, 0.5, 0.75, 1]
best_alpha = 0
best_l1_ratio = 0

for i in range(0, len(alpha_values)):
    for j in range(0, len(l1ratioValues)):
        best_rmse, best_alpha, best_l1_ratio = perform_elastic_net_regression(
            X_train, X_test, y_train, y_test,
            alpha_values[i], l1ratioValues[j], best_rmse,
            best_alpha, best_l1_ratio)

print("Best RMSE " + str(best_rmse) + " Best alpha: " + str(best_alpha)
      + "  " + "Best l1 ratio: " + str(best_l1_ratio))
