import pandas as pd
import os

from sklearn import metrics
import statsmodels.api as sm
import numpy as np

# Read the data:
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split

# Read the data:
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "USA_housing.csv"
df = pd.read_csv(PATH + CSV_DATA)

# Remove address column
del df['Address']

# Separate the target and independent variable
X = df.copy()     # Create separate copy to prevent unwanted tampering of data.
del X['Price']  # Delete target variable.

# Target variable
y = df['Price']

#  f_regression returns F statistic for each feature.
ffs = f_regression(X, y)

features_df = pd.DataFrame()
for i in range(0, len(X.columns)):
    features_df = features_df._append({"feature": X.columns[i],
                                       "ffs": ffs[0][i]}, ignore_index=True)
features_df = features_df.sort_values(by=['ffs'])
print(features_df)

# Now we can build our model with forward selected features.
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Area Population']]

# Adding an intercept *** This is required ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)  # make the predictions by the model

print(model.summary())

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
