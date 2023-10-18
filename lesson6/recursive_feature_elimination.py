import warnings
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import os

warnings.simplefilter(action="ignore", category=FutureWarning)

# Read the data:
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
df = pd.read_csv(PATH + "Divorce.csv", header=0)

# Separate the target and independent variable
X = df.copy()  # Create separate copy to prevent unwanted tampering of data.
del X['Divorce']  # Delete target variable.

# Target variable
y = df['Divorce']

# Create the object of the model
model = LogisticRegression()

# Specify the number of  features to select
rfe = RFE(model, n_features_to_select=8)

# fit the model
rfe = rfe.fit(X, y)

# Please uncomment the following lines to see the result
print('\n\nFEATURES SELECTED\n\n')
print(rfe.support_)

# Show top features.
for i in range(0, len(X.keys())):
    if rfe.support_[i]:
        print(X.keys()[i])
