import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE = "4949_assignmentData.csv"
df = pd.read_csv(PATH + FILE, parse_dates=['Date'], index_col='Date')

new_data = pd.DataFrame({'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'O': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'U': 0, 'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0}, index=[pd.to_datetime('2023-05-04')])
df = df._append(new_data)

for i in [1, 7, 8, 14]:
    df[f'A_lag_{i}'] = df['A'].shift(i)

df['C_lag_21'] = df['C'].shift(21)
df['D_lag_21'] = df['D'].shift(21)
df['E_lag_7'] = df['E'].shift(7)
df['F_lag_21'] = df['F'].shift(21)
df['H_lag_7'] = df['H'].shift(7)
df['I_lag_21'] = df['I'].shift(21)
df['J_lag_7'] = df['J'].shift(7)
df['L_lag_14'] = df['L'].shift(14)
df['L_lag_21'] = df['L'].shift(21)
df['M_lag_21'] = df['M'].shift(21)
df['M_lag_22'] = df['M'].shift(22)
df['P_lag_22'] = df['P'].shift(22)
df['Q_lag_2'] = df['Q'].shift(2)
df['Q_lag_21'] = df['Q'].shift(21)
df['S_lag_2'] = df['S'].shift(2)
df['S_lag_7'] = df['S'].shift(7)
df['U_lag_7'] = df['U'].shift(7)
df['U_lag_22'] = df['U'].shift(22)
df['W_lag_21'] = df['W'].shift(21)
df['Z_lag_2'] = df['Z'].shift(2)

# Impute missing values using KNNImputer.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)

X = df.drop('A', axis=1)
y = df['A']

# Drop all letters from B to Z
X = X.drop(['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], axis=1)

# Add intercept for OLS regression.
X = sm.add_constant(X)
TEST_DAYS = 10 + 1

# Split into test and train sets. The test data includes
# the latest values in the data.
len_data = len(X)
X_train = X[0:len_data - TEST_DAYS]
y_train = y[0:len_data - TEST_DAYS]
X_test = X[len_data - TEST_DAYS: -1]
y_test = y[len_data - TEST_DAYS: -1]

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
plt.title("Predicted values of A")
plt.tight_layout()
plt.show()

# Save model into a pickle
import pickle
with open(f"{ROOT_PATH}\\model1.pkl", 'wb') as file:
    pickle.dump(model, file)

# Predict the last row of data
new_data = X.iloc[-1]

# Show the last row of data with filled values
print("Last row of data (with timeshifted values):")
print(new_data)

new_data = new_data.values.reshape(1, -1)
print("Prediction:")
print(model.predict(new_data)[0])
