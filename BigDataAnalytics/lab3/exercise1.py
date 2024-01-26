import warnings

warnings.filterwarnings("ignore")

import statsmodels.api as sm

dta = sm.datasets.sunspots.load_pandas().data

import datetime
from pandas_datareader import data as pdr
import yfinance as yfin
import statsmodels.tsa.arima.model as sma
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np


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


stk_name = 'MSFT'
df_stock = get_stock(stk_name, 400)

# Take only the open price
df_stock = df_stock[['Open']]

# Split the data.
NUM_TEST_DAYS = 5
lenData = len(df_stock)
train = df_stock.iloc[0:lenData - NUM_TEST_DAYS, :]
test = df_stock.iloc[lenData - NUM_TEST_DAYS:, :]

import warnings

warnings.filterwarnings("ignore")


def build_model(df, ar, i, ma):
    model = sma.ARIMA(df['Open'], order=(ar, i, ma)).fit()
    return model


def predict_and_evaluate(model, test, title):
    print("\n***" + title)
    print(model.summary())
    start = len(train)
    end = start + len(test) - 1
    predictions = model.predict(start=start, end=end, dynamic=True)

    mse = mean_squared_error(predictions, test)

    rmse = np.sqrt(mse)
    print("RMSE: " + str(rmse))
    return rmse, predictions


def show_predicted_and_actual(actual, predictions, ar, ma):
    indices = list(actual.index)
    plt.title("AR: " + str(ar) + " MA: " + str(ma))
    plt.plot(indices, predictions, label='predictions', marker='o')
    plt.plot(indices, actual, label='actual', marker='o')
    plt.legend()
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()


model_stats = []
for ar in range(0, 5):
    for ma in range(0, 5):
        model = build_model(train, ar, 0, ma)
        title = str(ar) + "_0_" + str(ma)
        rmse, predictions = predict_and_evaluate(model, test, title)
        if ar == 2 and ma == 3:
            show_predicted_and_actual(test, predictions, ar, ma)
        model_stats.append({"ar": ar, "ma": ma, "rmse": rmse})

df_solutions = pd.DataFrame(data=model_stats)
df_solutions = df_solutions.sort_values(by=['rmse'])
print(df_solutions)
