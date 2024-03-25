# Read pickle with model1.pkl
import pickle
import os
import pandas as pd

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE = "4949_assignmentData.csv"

df = pd.read_csv(PATH + FILE, parse_dates=['Date'], index_col='Date')

# Add new row representing May 4th, 2023
new_data = pd.DataFrame({'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'O': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'U': 0, 'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0}, index=[pd.to_datetime('2023-05-04')])
df = df._append(new_data)

# Add lagged columns
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

# Add constant as the first column
df.insert(0, 'const', 1.00)

# Drop columns from A to Z (invalid for prediction)
df = df.drop(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], axis=1)

with open(f"{ROOT_PATH}\\model1.pkl", 'rb') as file:
    model = pickle.load(file)

# Predict the last row of the dataset
new_data = df.iloc[-1].values.reshape(1, -1)

print("New data:")
print(new_data)

print("Prediction:")
print(model.predict(new_data)[0])
