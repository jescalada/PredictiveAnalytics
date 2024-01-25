import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE = 'AirPassengers.csv'
forecast_data = pd.read_csv(PATH + FILE, index_col='date', parse_dates=True)
forecast_data.index.freq = 'MS'

# Split into train and test set.
train_airline = forecast_data[:120]
test_airline = forecast_data[120:]

# Build HWES3 model with multiplicative decomposition.
fitted_model = ExponentialSmoothing(train_airline['value'],
                                    trend='mul', seasonal='mul', seasonal_periods=12).fit()
test_predictions = fitted_model.forecast(24)

# Plot raw train, test and predictions.
train_airline['value'].plot(legend=True, label='TRAIN')
test_airline['value'].plot(legend=True, label='TEST', figsize=(6, 4))
test_predictions.plot(legend=True, label='PREDICTION')
plt.title('Train, Test and Predicted Test using Holt Winters')
plt.show()

# Show RMSE.
mse = mean_squared_error(test_airline, test_predictions)
rmse = np.sqrt(mse)
print('RMSE: ' + str(rmse))
