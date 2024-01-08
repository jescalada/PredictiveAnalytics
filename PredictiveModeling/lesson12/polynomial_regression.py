import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os

# The data file path and file name need to be configured.
CSV_DATA = "salaries.csv"

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = ROOT_PATH + "\\..\\datasets\\"
# Note this has a comma separator.
df = pd.read_csv(PATH + CSV_DATA)

# Extract target and salary.
x = df[['Level']]
y = df[['Salary']]

# Generate the matrix of x values which includes:
# x^0 (which equals 1), x^1, X^2, X^3 and X^4
polyFeatures = PolynomialFeatures(degree=3)
x_transformed = polyFeatures.fit_transform(x)

print("x_transformed:")
print(x_transformed)

# Perform linear regression with the transformed data.
linReg = LinearRegression()
linReg.fit(x_transformed, y)

print("Coefficients:")
print(linReg.coef_)

# Estimate dependent variable for polynomial equations.
predictions = linReg.predict(x_transformed)

# Visualize result for Polynomial Regression.
plt.scatter(x, y, color="blue")
plt.plot(x, predictions, color="red")
plt.title("Salary Prediction")
plt.xlabel("Position Levels")
plt.ylabel("Salary")
plt.show()

print(linReg.intercept_)
print(linReg.coef_)
