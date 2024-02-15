import math

from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE = "4949_assignmentData.csv"
df = pd.read_csv(PATH + FILE, parse_dates=['Date'], index_col='Date')

# Perform a simple exploratory data analysis.
# Expand to see all columns.
pd.set_option('display.max_columns', None)
print(df.head())
print(df.info())
print(df.describe())
print(df.index)
print(df.columns)

# Check if there are correlated columns.
print(df.corr())

# Add extra columns for the value of A in the previous 3 days.
for i in range(1, 4):
    df[f'A_lag_{i}'] = df['A'].shift(i)

# Print out the columns with correlations that represent over 50% variance
correlation_matrix = df.corr()
print("Correlated columns:")
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > math.sqrt(0.6):
            colname = correlation_matrix.columns[i]
            print(f"{correlation_matrix.columns[j]} and {colname} have a correlation of {correlation_matrix.iloc[i, j]:.3f}")

# Plot the data over time
df['A'].plot()
plt.show()

# Perform PACF and ACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df['A'], lags=40)
plt.show()
plot_pacf(df['A'], lags=40)
plt.show()
