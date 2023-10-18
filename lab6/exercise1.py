import pandas as pd
import sklearn.feature_selection
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
import numpy as np

# Read the data:
from sklearn.model_selection import train_test_split

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "USA_housing.csv"
df = pd.read_csv(PATH + CSV_DATA)

# Remove the address column
del df['Address']

# Seperate the target and independent variable
X = df.copy()  # Create separate copy to prevent unwanted tampering of data.
del X['Price']  # Delete target variable.

# Target variable
y = df['Price']

# Create the object of the model
model = LinearRegression()

# Specify the number of  features to select
rfe = sklearn.feature_selection.RFE(model, n_features_to_select=4)

# fit the model
rfe = rfe.fit(X, y)
# Please uncomment the following lines to see the result
print('\n\nFEATURES SELECTED\n\n')

columns = list(X.keys())
for i in range(0, len(columns)):
    if (rfe.support_[i]):
        print(columns[i])

# Now we can build our model with forward selected features.
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms']]

# Adding an intercept *** This is required ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)  # make the predictions by the model

print(model.summary())

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
