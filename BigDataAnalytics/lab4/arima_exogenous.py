import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yfin  # Work around until
# pandas_datareader is fixed.
import datetime
import pmdarima as pm
from sklearn.metrics import mean_squared_error
from math import sqrt

warnings.filterwarnings("ignore")


def get_stock(stk, ttl_days):
    num_days = int(ttl_days)
    # Only gets up until day before during
    # trading hours
    dt = datetime.date.today()
    # For some reason, must add 1 day to get current stock prices
    # during trade hours. (Prices are about 15 min behind actual prices.)
    dt_now = dt + datetime.timedelta(days=1)
    dt_now_str = dt_now.strftime("%Y-%m-%d")
    dt_past = dt + datetime.timedelta(days=-num_days)
    dt_past_str = dt_past.strftime("%Y-%m-%d")
    yfin.pdr_override()
    df = pdr.get_data_yahoo(stk, start=dt_past_str, end=dt_now_str)
    return df


# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

dfStock = get_stock('AAPL', 1200)
print(dfStock)

TEST_DAYS = 5

# Build feature set with backshifted closing prices.
dfStock['Close_t_1'] = dfStock['Close'].shift(1)
dfStock = dfStock.dropna()
dfX = dfStock[['Open', 'Close_t_1']]
size = len(dfX) - TEST_DAYS

train = dfStock[0:len(dfStock) - TEST_DAYS]
test = dfStock[len(dfStock) - TEST_DAYS:]

train, test = dfX[0:size], dfX[size:]
predictions = []

# Iterate to make predictions for the evaluation set.
for i in range(0, len(test)):
    print("\n\nModel " + str(i))

    model = pm.auto_arima(train[['Open']],
                          exogenous=train[['Close_t_1']],
                          start_p=1, start_q=1,
                          test='adf',  # Use adftest to find optimal 'd'
                          max_p=3, max_q=3,  # Set maximum p and q.
                          d=None,  # Let model determine 'd'.
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True)

    lenOpen = len(train[['Close_t_1']])
    yhat, confint = model.predict(n_periods=1,
                                  exogenous=np.array(
                                      train.iloc[lenOpen - 1]['Close_t_1']).reshape(1, -1),
                                  return_conf_int=True)
    predictions.append(yhat)
    open = test.iloc[i]['Open']
    close_t_1 = test.iloc[i]['Close_t_1']
    train = train._append({"Open": open, "Close_t_1": close_t_1},
                          ignore_index=True)

plt.plot(test.index, test['Open'], marker='o',
         label='Actual', color='blue')
plt.plot(test.index, predictions, marker='o',
         label='Predicted', color='orange')
plt.legend()
plt.xticks(rotation=70)
plt.show()

rmse = sqrt(mean_squared_error(test['Open'], predictions))
print('Test RMSE: %.3f' % rmse)
