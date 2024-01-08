import pandas as pd
from sklearn.feature_selection import RFE
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
import numpy as np


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


# Now we can build our model with forward selected features.
X = df[['volatile acidity', 'chlorides', 'density',
        'pH', 'sulphates']]

# Adding an intercept *** This is required ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)  # make the predictions by the model

print(model.summary())

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
