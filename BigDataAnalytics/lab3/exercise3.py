import warnings
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

# Import data
df = pd.read_csv(
    "https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv", \
    names=['value'], header=0)
print(df)
df.value.plot()
plt.title("www usage")
plt.show()

result = adfuller(df.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# Difference the data.
df_diff = df.diff()

# Plot the differenced data
df_diff.plot()
plt.title("www usage differenced")
plt.show()

# Perform the ADF test again
result = adfuller(df_diff.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# Difference again.
df_diff = df_diff.diff()

# Plot the differenced data
df_diff.plot()
plt.title("www usage differenced twice")
plt.show()

# Perform the ADF test again
result = adfuller(df_diff.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
