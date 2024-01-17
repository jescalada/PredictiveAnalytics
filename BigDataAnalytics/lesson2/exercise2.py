from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE = "AirPassengers.csv"
df = pd.read_csv(PATH + FILE, parse_dates=['date'], index_col='date')
type(df.index)

# Perform decomposition using multiplicative decomposition.
tseries_multiplicative = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend="freq")

tseries_multiplicative.plot()

# Extract the Components ----
# Actual Values = Product of (Seasonal * Trend * Resid)
dfComponents = pd.concat([tseries_multiplicative.seasonal, tseries_multiplicative.trend,
                          tseries_multiplicative.resid, tseries_multiplicative.observed], axis=1)
dfComponents.columns = ['seas', 'trend', 'resid', 'actual_values']
print(dfComponents.head())

tseries_additive = seasonal_decompose(df['value'], model='additive', extrapolate_trend="freq")

# Extract the Components ----
# Actual Values = Seasonal + Trend + Resid
dfComponents = pd.concat([tseries_additive.seasonal, tseries_additive.trend,
                            tseries_additive.resid, tseries_additive.observed], axis=1)
dfComponents.columns = ['seas', 'trend', 'resid', 'actual_values']
print(dfComponents.head())

# Plot both multiplicative and additive decompositions.
fig, axes = plt.subplots(8, 1, figsize=(10, 20), dpi=100, sharex=True)
tseries_multiplicative.observed.plot(ax=axes[0], legend=False, color='red')
axes[0].set_ylabel('Multiplicative Observed')
tseries_additive.observed.plot(ax=axes[1], legend=False, color='red')
axes[1].set_ylabel('Additive Observed')
tseries_multiplicative.trend.plot(ax=axes[2], legend=False, color='green')
axes[2].set_ylabel('Multiplicative Trend')
tseries_additive.trend.plot(ax=axes[3], legend=False, color='green')
axes[3].set_ylabel('Additive Trend')
tseries_multiplicative.seasonal.plot(ax=axes[4], legend=False, color='blue')
axes[4].set_ylabel('Multiplicative Seasonal')
tseries_additive.seasonal.plot(ax=axes[5], legend=False, color='blue')
axes[5].set_ylabel('Additive Seasonal')
tseries_multiplicative.resid.plot(ax=axes[6], legend=False, color='black')
axes[6].set_ylabel('Multiplicative Residual')
tseries_additive.resid.plot(ax=axes[7], legend=False, color='black')
axes[7].set_ylabel('Additive Residual')
plt.tight_layout()
plt.show()
