import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import helpers

ROOT_DATA = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "\\..\\datasets\\"
FILE = 'loan_v2.csv'
df = pd.read_csv(ROOT_DATA + DATASET_DIR + FILE)

# Create X from the features.
X = df[['Income (USD)', 'Loan Amount Request (USD)', 'Current Loan Expenses (USD)', 'Credit Score']]
X = sm.add_constant(X)

# Replace invalid values with 0
X = X.replace(np.nan, 0, regex=True)

# Clip negative values to 0
X = X.clip(lower=0)

# Scale data using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Create y from output.
y = df['Loan Sanction Amount (USD)']

helpers.validate_and_evaluate(X, y, k=20)
