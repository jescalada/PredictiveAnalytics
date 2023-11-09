import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
import helpers

ROOT_DATA = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "\\..\\datasets\\"
FILE = 'loan_v2.csv'
df = pd.read_csv(ROOT_DATA + DATASET_DIR + FILE)

X = df.copy()     # Create separate copy to prevent unwanted tampering of data.
del X['Loan Sanction Amount (USD)']  # Delete target variable.

# Target variable
y = df['Loan Sanction Amount (USD)']

# Convert categorical data to numerical data.
X = pd.get_dummies(X)

# Impute missing values with KNN
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=4)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Replace invalid values with 0
X = X.replace(np.nan, 0, regex=True)

# Replace False with 0 and True with 1
X = X.replace(False, 0, regex=True)
X = X.replace(True, 1, regex=True)

# Replace negative values with 0
X = X.clip(lower=0)

#  f_regression returns F statistic for each feature.
ffs = f_regression(X, y)

features_df = pd.DataFrame()
for i in range(0, len(X.columns)):
    features_df = features_df._append({"feature": X.columns[i],
                                       "ffs": ffs[0][i]}, ignore_index=True)
features_df = features_df.sort_values(by=['ffs'])

# Select features with F statistic greater than 50.
selected = [x if y > 50 else "" for [x, y] in features_df.values.tolist()]

# Remove empty strings from list.
selected = list(filter(None, selected))

# Create X from the features.
X_selected_features = X[selected]
X_selected_features = sm.add_constant(X_selected_features)

# Create y from output.
y = df['Loan Sanction Amount (USD)']

helpers.validate_and_evaluate(X_selected_features, y, k=5)
