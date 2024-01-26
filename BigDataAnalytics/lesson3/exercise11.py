from pandas import read_csv
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import warnings
import os

warnings.filterwarnings("ignore")

# Load the data.
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
series = read_csv(PATH + 'daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True)

# Plot ACF.
plot_acf(series, lags=20)
plt.show()

# Plot PACF.
plot_pacf(series, lags=20)

plt.show()
NUM_TEST_DAYS = 7

# Split dataset into test and train.
X = series.values
lenData = len(X)
train = X[0:lenData - NUM_TEST_DAYS]
test = X[lenData - NUM_TEST_DAYS:]

# Train.
model = AutoReg(train, lags=7)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)

print(model_fit.summary())

# Make predictions.
predictions = model_fit.predict(start=len(train),
                                end=len(train) + len(test) - 1,
                                dynamic=False)

for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# Plot results.
plt.plot(test, marker='o', label='actual')
plt.plot(predictions, color='brown', linewidth=4,
         marker='o', label='predicted')

plt.legend()
plt.show()


# Use model coefficients from autoregression to make a prediction.
def make_prediction(t_1, t_2, t_3, t_4, t_5, t_6, t_7):
    intercept = 1.11532391
    t1_coeff = 0.62644214
    t2_coeff = -0.07506915
    t3_coeff = 0.07390916
    t4_coeff = 0.06186014
    t5_coeff = 0.06587204
    t6_coeff = 0.04415531
    t7_coeff = 0.10268948

    prediction = intercept + t1_coeff * t_1 \
                 + t2_coeff * t_2 \
                 + t3_coeff * t_3 \
                 + t4_coeff * t_4 \
                 + t5_coeff * t_5 \
                 + t6_coeff * t_6 \
                 + t7_coeff * t_7
    return prediction


test_len = len(test)

t_1 = test[test_len - 1]
t_2 = test[test_len - 2]
t_3 = test[test_len - 3]
t_4 = test[test_len - 4]
t_5 = test[test_len - 5]
t_6 = test[test_len - 6]
t_7 = test[test_len - 7]

futurePredictions = []
for i in range(0, NUM_TEST_DAYS):
    prediction = make_prediction(t_1, t_2, t_3, t_4, t_5, t_6, t_7)
    futurePredictions.append(prediction)
    t_7 = t_6
    t_6 = t_5
    t_5 = t_4
    t_4 = t_3
    t_3 = t_2
    t_2 = t_1
    t_1 = prediction

print("Here is a one week temperature forecast: ")
print(futurePredictions)
