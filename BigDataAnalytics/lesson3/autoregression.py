import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings

warnings.filterwarnings("ignore")

dta = sm.datasets.sunspots.load_pandas().data
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))

# Show autocorrelation function.
# General correlation of lags with past lags.
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(dta['SUNACTIVITY'], lags=50)
plt.show()

# Split the data.
NUM_TEST_YEARS = 10
lenData = len(dta)
df_train = dta.iloc[0:lenData - NUM_TEST_YEARS, :]
df_test = dta.iloc[lenData - NUM_TEST_YEARS:, :]


def build_model_and_make_predictions(AR_time_steps, df_train, df_test):
    # This week we will use the ARIMA model.

    model = ARIMA(df_train['SUNACTIVITY'], order=(AR_time_steps, 0, 0),
                  freq='Y').fit()
    print("\n*** Evaluating ARMA(" + str(AR_time_steps) + ",0,0)")
    print('Coefficients: %s' % model.params)

    # Strings which can be converted to time stamps are passed in.
    # For this case the entire time range for the test set is represented.
    predictions = model.predict('1999-12-31', '2008-12-31', dynamic=True)
    rmse = np.sqrt(mean_squared_error(df_test['SUNACTIVITY'].values,
                                      np.array(predictions)))
    print('Test RMSE: %.3f' % rmse)
    print('Model AIC %.3f' % model.aic)
    print('Model BIC %.3f' % model.bic)
    return model, predictions


print(df_test)
arma_mod20, predictionsARMA_20 = build_model_and_make_predictions(2, df_train, df_test)
arma_mod30, predictionsARMA_30 = build_model_and_make_predictions(3, df_train, df_test)
arma_mod90, predictionsARMA_90 = build_model_and_make_predictions(9, df_train, df_test)

plt.plot(df_test.index, df_test['SUNACTIVITY'],
         label='Actual Values', color='blue')
plt.plot(df_test.index, predictionsARMA_20,
         label='Predicted Values AR(20)', color='orange')
plt.plot(df_test.index, predictionsARMA_30,
         label='Predicted Values AR(30)', color='brown')
plt.plot(df_test.index, predictionsARMA_90,
         label='Predicted Values AR(90)', color='green')

plt.legend(loc='best')
plt.show()

print("\n*** Comparing AR Model Time Steps")
import warnings
warnings.filterwarnings("ignore")
res = sm.tsa.arma_order_select_ic(dta['SUNACTIVITY'], max_ar=10, max_ma=0, ic='bic')
print(res.values())

# Print AIC, BIC and RMSE for AR(2), AR(3) and AR(9)
aic = [arma_mod20.aic, arma_mod30.aic, arma_mod90.aic]
bic = [arma_mod20.bic, arma_mod30.bic, arma_mod90.bic]
rmse = [np.sqrt(mean_squared_error(df_test['SUNACTIVITY'].values,
                                      np.array(predictionsARMA_20))),
        np.sqrt(mean_squared_error(df_test['SUNACTIVITY'].values,
                                        np.array(predictionsARMA_30))),
        np.sqrt(mean_squared_error(df_test['SUNACTIVITY'].values,
                                        np.array(predictionsARMA_90)))]
print("AIC: " + str(aic))
print("BIC: " + str(bic))
print("RMSE: " + str(rmse))