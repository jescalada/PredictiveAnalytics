import warnings

warnings.filterwarnings("ignore")

import statsmodels.api as sm
import statsmodels.tsa.arima.model as sma
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np


def get_data():
    df = sm.datasets.sunspots.load_pandas().data[['SUNACTIVITY']]
    df.index = pd.date_range(start='1700', end='2009', freq='A')
    TEST_SZ = 5
    train = df[0:len(df) - TEST_SZ]
    test = df[len(df) - TEST_SZ:]
    return train, test


def build_model(df, ar, i, ma):
    model = sma.ARIMA(df['SUNACTIVITY'], order=(ar, i, ma)).fit()
    return model


def predict_and_evaluate(model, test, title):
    print("\n***" + title)
    print(model.summary())
    predictions = model.predict(start='2004', end='2008')
    mse = mean_squared_error(predictions, test)

    rmse = np.sqrt(mse)
    print("RMSE: " + str(rmse))
    return rmse, predictions


train, test = get_data()


def show_predicted_and_actual(actual, predictions, ar, ma):
    indicies = list(actual.index)
    plt.title("AR: " + str(ar) + " MA: " + str(ma))
    plt.plot(indicies, predictions, label='predictions', marker='o')
    plt.plot(indicies, actual, label='actual', marker='o')
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
        if ar == 3 and ma == 0:
            show_predicted_and_actual(test, predictions, ar, ma)
        model_stats.append({"ar": ar, "ma": ma, "rmse": rmse})

df_solutions = pd.DataFrame(data=model_stats)
df_solutions = df_solutions.sort_values(by=['rmse'])
print(df_solutions)
