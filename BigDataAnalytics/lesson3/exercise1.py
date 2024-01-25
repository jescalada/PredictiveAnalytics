import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import numpy as np
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE = 'DailyDelhiClimateTest.csv'
forecast_data = pd.read_csv(PATH + FILE, index_col='date', parse_dates=True)
forecast_data.index.freq = 'D'

# Split into train and test set.
train_temps = forecast_data[:90]
test_temps = forecast_data[89:]

# Perform seasonal decomposition.
decompose_result = seasonal_decompose(train_temps['meantemp'], model='multiplicative')
decompose_result.plot()
plt.title('Seasonal Decomposition of Train Temps')
plt.show()

# Build HWES2 model with multiplicative decomposition.
fitted_model = ExponentialSmoothing(train_temps['meantemp'],
                                    trend='mul').fit()

# Predict next 24 values.
test_predictions = fitted_model.forecast(24)

# Plot raw train, test and predictions.
train_temps['meantemp'].plot(legend=True, label='TRAIN')
test_temps['meantemp'].plot(legend=True, label='TEST', figsize=(6, 4))
test_predictions.plot(legend=True, label='PREDICTION')
plt.title('Train, Test and Predicted Test Temps using Holt Winters (Double)')
plt.show()

# Plot raw train, test and predictions.
train_temps['meantemp'].plot(legend=True, label='TRAIN')
test_temps['meantemp'].plot(legend=True, label='TEST', figsize=(6, 4))
fitted_model.plot(legend=True, label='PREDICTION')
plt.title('Train, Test and Predicted Test Temps using Holt Winters (Double)')
plt.show()

# Build HWES3 model with multiplicative decomposition.
fitted_model = ExponentialSmoothing(train_temps['meantemp'],
                                    trend='mul', seasonal='mul', seasonal_periods=12).fit()
test_predictions = fitted_model.forecast(24)

# Plot raw train, test and predictions.
train_temps['meantemp'].plot(legend=True, label='TRAIN')
test_temps['meantemp'].plot(legend=True, label='TEST', figsize=(6, 4))
test_predictions.plot(legend=True, label='PREDICTION')
plt.title('Train, Test and Predicted Test Temps using Holt Winters (Triple)')
plt.show()

# Show RMSE.
mse = mean_squared_error(test_temps, test_predictions)
rmse = np.sqrt(mse)
print('RMSE: ' + str(rmse))
