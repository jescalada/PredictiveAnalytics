import warnings
import os

import matplotlib.pyplot as plt
import pmdarima as pm
import numpy as np

import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse


warnings.filterwarnings("ignore")

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE_NAME = "Energy_Production.csv"

# Import
data = pd.read_csv(PATH + FILE_NAME, parse_dates=['DATE'], index_col='DATE')

print(data.head())
print(data.describe())

# Rename column IPG2211A2N to value and DATE to date
data = data.rename(columns={'IPG2211A2N': 'value', 'DATE': 'date'})

# multiplicative seasonal component
result_mul = seasonal_decompose(data['value'][-36:],  # 3 years
                                model='multiplicative',
                                extrapolate_trend='freq')

seasonal_index = result_mul.seasonal[-12:].to_frame()
seasonal_index['month'] = pd.to_datetime(seasonal_index.index).month

# merge with the base data
data['month'] = data.index.month
df = pd.merge(data, seasonal_index, how='left', on='month')
df.columns = ['value', 'month', 'seasonal_index']
df.index = data.index  # reassign the index.

# SARIMAX Model
sxmodel = pm.auto_arima(df[['value']], exogenous=df[['seasonal_index']],
                        start_p=1, start_q=1,
                        test='adf',
                        max_p=3, max_q=3, m=12,
                        start_P=0, seasonal=True,
                        d=None, D=1, trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True)

sxmodel.summary()

n_periods = 24

# Forecast
n_periods = 24
fitted, confint = sxmodel.predict(n_periods=n_periods,
                                  exogenous=np.tile(seasonal_index.values, 2).reshape(-1, 1),
                                  return_conf_int=True)

index_of_fc = pd.date_range(data.index[-1], periods=n_periods, freq='MS')

# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(data['value'])
plt.plot(fitted_series, color='darkgreen')
plt.fill_between(lower_series.index,
                 lower_series,
                 upper_series,
                 color='k', alpha=.15)

plt.title("SARIMAX Forecast - Energy Production")
plt.show()

# Seasonal differencing

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(data[:], label='Original Series')
axes[0].plot(data[:].diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)

# Seasonal Differencing
axes[1].plot(data[:], label='Original Series')
axes[1].plot(data[:].diff(12), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.show()
