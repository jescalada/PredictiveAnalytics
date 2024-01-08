import pandas as pd
from sklearn.feature_selection import f_regression
import os

# Read the data:
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "winequality.csv"
df = pd.read_csv(PATH + CSV_DATA)

# Separate the target and independent variable
X = df.copy()     # Create separate copy to prevent unwanted tampering of data.
del X['quality']  # Delete target variable.

# Target variable
y = df['quality']

#  f_regression returns F statistic for each feature.
ffs = f_regression(X, y)

features_df = pd.DataFrame()
for i in range(0, len(X.columns)):
    features_df = features_df._append({"feature": X.columns[i],
                                       "ffs": ffs[0][i]}, ignore_index=True)
features_df = features_df.sort_values(by=['ffs'])
print(features_df)
