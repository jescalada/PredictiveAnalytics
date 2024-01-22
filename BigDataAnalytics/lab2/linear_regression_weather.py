import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from sklearn import metrics

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Do not show warning.
pd.options.mode.chained_assignment = None

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE = 'DailyDelhiClimateTest.csv'
df = pd.read_csv(PATH + FILE, parse_dates=['date'], index_col='date')
print(df)


# Create back-shifted columns for an attribute.
def add_back_shifted_columns(df, col_name, time_lags):
    for i in range(1, time_lags + 1):
        new_col_name = col_name + "_t-" + str(i)
        df[new_col_name] = df[col_name].shift(i)
    return df


# Build dataframe for modelling.
columns = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
model_df = df.copy()
NUM_TIME_STEPS = 3
for i in range(0, len(columns)):
    model_df = add_back_shifted_columns(model_df, columns[i],
                                        NUM_TIME_STEPS)
model_df = model_df.dropna()
y = model_df[['meantemp']]
X = model_df[['meantemp_t-1']]

# Add intercept for OLS regression.
X = sm.add_constant(X)
TEST_DAYS = 10

# Split into test and train sets. The test data includes
# the latest values in the data.
len_data = len(X)
X_train = X[0:len_data - TEST_DAYS]
y_train = y[0:len_data - TEST_DAYS]
X_test = X[len_data - TEST_DAYS:]
y_test = y[len_data - TEST_DAYS:]

# Model and make predictions.
model = sm.OLS(y_train, X_train).fit()
print(model.summary())
predictions = model.predict(X_test)

print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Plot the data.
xaxis_values = list(y_test.index)
plt.plot(xaxis_values, y_test, label='Actual', marker='o')
plt.plot(xaxis_values, predictions, label='Predicted', marker='o')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.title("Mean temperature in Dehli")
plt.show()
