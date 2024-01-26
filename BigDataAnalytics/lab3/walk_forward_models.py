from pandas import read_csv
import matplotlib.pyplot as plt
import statsmodels.tsa.arima.model as sma
from sklearn.metrics import mean_squared_error
from math import sqrt
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
df = read_csv(PATH + 'daily-min-temperatures.csv', header=0, index_col=0)

TEST_DAYS = 5

X_train = df[0:len(df)-TEST_DAYS]
y_train = df[0:len(df)-TEST_DAYS]
X_test  = df[len(df)-TEST_DAYS:]
y_test  = df[len(df)-TEST_DAYS:]

# Create a list with the training array.
predictions = []

for i in range(len(X_train)):
    print("History length: " + str(len(X_train)))

    #################################################################
    # Model building and prediction section.
    model       = sma.ARIMA(X_train, order=(1, 0, 0)).fit()
    yhat        = model.predict(start=len(X_train), end=len(X_train))

    if i < len(X_test):
        test_row = X_test.iloc[i]
        X_train = X_train._append(test_row, ignore_index=True )
        predictions.append(yhat.iloc[0])
    else:
        break

    #################################################################

rmse = sqrt(mean_squared_error(X_test, predictions))
print('Test RMSE: %.3f' % rmse)

indices = list(X_test.index)
plt.plot(indices, X_test, label='Actual', marker='o', color='blue')
plt.plot(indices, predictions, label='Predictions', marker='o', color='orange')
plt.legend()
plt.title("AR Model")
plt.xticks(rotation=70)
plt.tight_layout()
plt.show()
