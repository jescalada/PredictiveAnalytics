import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import os

# Read the data:
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "winequality.csv"
df = pd.read_csv(PATH + CSV_DATA)

# Seperate the target and independent variable
X = df.copy()  # Create separate copy to prevent unwanted tampering of data.
del X['quality']  # Delete target variable.

# Target variable
y = df['quality']

# Create the object of the model
model = LinearRegression()

# Specify the number of  features to select
rfe = RFE(model, n_features_to_select=5)

# fit the model
rfe = rfe.fit(X, y)
# Please uncomment the following lines to see the result
print('\n\nFEATURES SELECTED\n\n')
print(rfe.support_)

columns = list(X.keys())
for i in range(0, len(columns)):
    if (rfe.support_[i]):
        print(columns[i])
