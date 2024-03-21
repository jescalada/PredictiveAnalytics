import pandas as pd
from matplotlib import pyplot as plt
import os
from sklearn.metrics import mean_squared_error
import numpy as np

from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE = '4949_assignmentData.csv'
df = pd.read_csv(PATH + FILE, index_col='Date', parse_dates=['Date'])

# Split into train and test set
train = df.iloc[:len(df) - 10]
test = df.iloc[len(df) - 10:]

# Build HWES2 model with multiplicative decomposition.
fitted_model = ExponentialSmoothing(train['A'],
                                    trend='mul', seasonal='mul', seasonal_periods=12).fit()

# Predict next 10 values.
test_predictions = fitted_model.forecast(10)

# Set the dates for the next 10 values
test_predictions.index = pd.date_range(start=train.index[-1], periods=10, freq='D')[0:]

# Plot raw train, test and predictions.
train['A'].plot(legend=True, label='TRAIN')
test['A'].plot(legend=True, label='TEST', figsize=(6, 4))
test_predictions.plot(legend=True, label='PREDICTION')
plt.title('Train, Test and Predicted A values using Holt Winters (Double)')
plt.show()

# Calculate RMSE
mse = mean_squared_error(test['A'], test_predictions)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')
