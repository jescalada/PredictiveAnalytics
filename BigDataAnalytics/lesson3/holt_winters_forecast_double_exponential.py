import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os

# holt winters
# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE = 'AirPassengers.csv'
airline = pd.read_csv(PATH + FILE, index_col='date', parse_dates=True)
# finding shape of the dataframe
print(airline.shape)
# having a look at the data
print(airline.head())
# plotting the original data
airline[['value']].plot(title='Passengers Data')
decompose_result = seasonal_decompose(airline['value'],model='multiplicative')
decompose_result.plot()
plt.show()

# Set the frequency of the date time index as Monthly start as indicated by the data
airline.index.freq = 'MS'
# Set the value of Alpha and define m (Time Period)
m = 12
alpha = 1/(2*m)
airline['HWES1'] = SimpleExpSmoothing(airline['value']).fit(smoothing_level=alpha, optimized=False, use_brute=True ).fittedvalues
airline[['value','HWES1']].plot(title='Holt Winters Single Exponential Smoothing')
plt.show()
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os

# holt winters
# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE = 'AirPassengers.csv'
airline = pd.read_csv(PATH + FILE, index_col='date', parse_dates=True)
# finding shape of the dataframe
print(airline.shape)
# having a look at the data
print(airline.head())
# plotting the original data
airline[['value']].plot(title='Passengers Data')
decompose_result = seasonal_decompose(airline['value'],model='multiplicative')
decompose_result.plot()
plt.show()

# HWES2 model fit with raw and additive trend data.
airline['HWES2_ADD'] = ExponentialSmoothing(airline['value'],trend='add').fit().fittedvalues

# HWES2 model fit with raw and multiplicative trend data.
airline['HWES2_MUL'] = ExponentialSmoothing(airline['value'],trend='mul').fit().fittedvalues

# Plot raw data as well as fitted values for both HWES2 models.
airline[['value','HWES2_ADD','HWES2_MUL']].plot(title=\
                    'Holt Winters Double Exponential Smoothing: Additive and Multiplicative Trend')
plt.show()
