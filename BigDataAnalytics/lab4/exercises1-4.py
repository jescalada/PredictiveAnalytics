import warnings

warnings.filterwarnings("ignore")

from pandas import read_csv
import matplotlib.pyplot as plt
import statsmodels.tsa.arima.model as sma
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import os
import pmdarima as pm

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
df = read_csv(PATH + 'daily-total-female-births.csv',
              header=0, index_col=0)

print(df.head())
print(df.describe())

# Split the data set so the test set is 7.
TEST_DAYS = 7

X_train = df[0:len(df) - TEST_DAYS]
y_train = df[0:len(df) - TEST_DAYS]
X_test = df[len(df) - TEST_DAYS:]
y_test = df[len(df) - TEST_DAYS:]

# Create a list with the training array.
predictions = []

for i in range(len(X_test)):
    print("History length: " + str(len(X_train)))

    print("\n*****************************************")
    print("Iteration: " + str(i))
    print("Length of training data: " + str(len(X_train)))
    model = pm.auto_arima(X_train, start_p=1, start_q=1,
                          test='adf',
                          max_p=3, max_q=3, m=0,
                          start_P=0, seasonal=False,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
    yhat = model.predict(start=len(X_train), end=len(X_train))

    if (i < len(X_test)):
        test_row = X_test.iloc[i]
        X_train = X_train._append(test_row, ignore_index=True)
        predictions.append(yhat.iloc[0])
    else:
        break


    print(model.summary())


rmse = sqrt(mean_squared_error(X_test, predictions))
print('Test RMSE: %.3f' % rmse)

plt.plot(X_test, label='Actual', marker='o', color='blue')
plt.plot(predictions, label='Predictions', marker='o', color='orange')
plt.legend()
plt.title("AR Model")
plt.show()
