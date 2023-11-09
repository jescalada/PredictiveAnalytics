import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import helpers

### Load Data ###
ROOT_DATA = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "\\..\\datasets\\"
FILE = 'loan_v2.csv'
df = pd.read_csv(ROOT_DATA + DATASET_DIR + FILE)

X = df.copy()     # Create separate copy to prevent unwanted tampering of data.
del X['Loan Sanction Amount (USD)']  # Delete target variable.

# Target variable
y = df['Loan Sanction Amount (USD)']

### Feature Engineering ###
# Add feature for age 65
X['Age 65'] = X['Age'] >= 65

# Set age values to NaN if they are 18 or less
X.loc[X['Age'] <= 18, 'Age'] = np.nan

# Convert categorical data to numerical data.
X = pd.get_dummies(X)

### Impute Missing Values ###
# Impute missing values in Income column with KNN
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Replace invalid values with 0
X = X.replace(np.nan, 0, regex=True)

# Replace False with 0 and True with 1
X = X.replace(False, 0, regex=True)
X = X.replace(True, 1, regex=True)

### Clip Values ###
# Replace negative values with 0
X = X.clip(lower=0)

# Clip income to 0-10000
UPPER_INCOME = 10000
X['Income (USD)'] = X['Income (USD)'].clip(lower=0, upper=UPPER_INCOME)

# Clip requested loan amount
UPPER_LOAN_REQUEST = 420000
X['Loan Amount Request (USD)'] = X['Loan Amount Request (USD)'].clip(lower=0, upper=UPPER_LOAN_REQUEST)

# Clip current loan expenses
UPPER_CURRENT_LOAN_EXPENSES = 1500

### Binning ###
# Create bins for credit score
X['Credit Score Bin'] = pd.cut(x=X['Credit Score'], bins=[580, 620, 650, 700, 750, 800, 850, 900])
temp_X = X[['Credit Score Bin']] # Isolate columns

# Get dummies as 0/1 columns
dummy_X = pd.get_dummies(temp_X, columns=['Credit Score Bin'])

# Replace True/False with 1/0
dummy_X = dummy_X.replace(True, 1, regex=True)
dummy_X = dummy_X.replace(False, 0, regex=True)

# Remove original column and join dummy_x to the dataframe
del X['Credit Score Bin']
X = pd.concat(([X, dummy_X]), axis=1)  # Join dummy df with original

# Create bins for loan amount request
X['Loan Amount Request Bin'] = pd.cut(x=X['Loan Amount Request (USD)'], bins=[0, 15000, 25000, 35000, 55000, 80000, 125000, UPPER_LOAN_REQUEST])

temp_X = X[['Loan Amount Request Bin']] # Isolate columns

# Get dummies as 0/1 columns
dummy_X = pd.get_dummies(temp_X, columns=['Loan Amount Request Bin'])

# Replace True/False with 1/0
dummy_X = dummy_X.replace(True, 1, regex=True)
dummy_X = dummy_X.replace(False, 0, regex=True)

# Remove original column and join dummy_x to the dataframe
del X['Loan Amount Request Bin']
X = pd.concat(([X, dummy_X]), axis=1)  # Join dummy df with original

# Create bins for Current Loan Expenses
X['Current Loan Expenses (USD) Bin'] = pd.cut(x=X['Current Loan Expenses (USD)'], bins=[0, 150, 250, 350, 450, 550, 700, 1000, UPPER_CURRENT_LOAN_EXPENSES])

temp_X = X[['Current Loan Expenses (USD) Bin']] # Isolate columns

# Get dummies as 0/1 columns
dummy_X = pd.get_dummies(temp_X, columns=['Current Loan Expenses (USD) Bin'])

# Replace True/False with 1/0
dummy_X = dummy_X.replace(True, 1, regex=True)
dummy_X = dummy_X.replace(False, 0, regex=True)

# Remove original column and join dummy_x to the dataframe
del X['Current Loan Expenses (USD) Bin']
X = pd.concat(([X, dummy_X]), axis=1)  # Join dummy df with original

### Feature Selection ###
# Drop features with low t-values (based on previous executions)
X_selected_features = X.drop(
    columns=['Age', 'Income (USD)', 'Credit Score', 'Property Age', 'Property Type', 'Property ID', 'Gender_F',
             'Gender_M', 'Profession_Businessman', 'Profession_Commercial associate', 'Profession_State servant',
             'Profession_Working', 'Type of Employment_Accountants', 'Type of Employment_Cleaning staff',
             'Type of Employment_Cooking staff', 'Type of Employment_Core staff', 'Type of Employment_HR staff',
             'Type of Employment_High skill tech staff', 'Type of Employment_IT staff',
             'Type of Employment_Low-skill Laborers', 'Type of Employment_Medicine staff',
             'Type of Employment_Private service staff', 'Type of Employment_Realty agents',
             'Type of Employment_Secretaries', 'Type of Employment_Security staff',
             'Type of Employment_Waiters/barmen staff', 'Type of Employment_Sales staff',
             'Location_Rural', 'Location_Semi-Urban', 'Location_Urban', 'Has Active Credit Card_Active',
             'Has Active Credit Card_Inactive', 'Has Active Credit Card_Unpossessed',
             'Loan Amount Request Bin_(35000, 55000]'])

# Add constant to X
X_selected_features = sm.add_constant(X_selected_features)

# Create y from output.
y = df['Loan Sanction Amount (USD)']

best_model = helpers.validate_and_evaluate(X_selected_features, y, 5)

# Get test data
X_train, X_test, y_train, y_test = train_test_split(X_selected_features, y, test_size=0.2)

# Plot predictions vs actual
helpers.plot_prediction_vs_actual(best_model, X_test, y_test, ": Model 3")

# Plot residuals
helpers.plot_residuals_vs_actual(best_model, X_test, y_test, ": Model 3")

equation = "Loan Sanction Amount (USD) = "
for x, y in dict(best_model.params).items():
    if x == 'const':
        equation += str(round(y, 4))
    else:
        equation += " + " + str(round(y, 4)) + " * " + x
print(equation)

python_equation = "return "
for x, y in dict(best_model.params).items():
    if x == 'const':
        # Add sign to constant
        python_equation += str(round(y, 4)) if y < 0 else "+" + str(round(y, 4))
    else:
        # Convert variable names to snake case
        x = x.replace(" ", "_")
        x = x.replace("(", "")
        x = x.replace(")", "")
        python_equation += " + " + str(round(y, 4)) + " * " + x
print(python_equation)