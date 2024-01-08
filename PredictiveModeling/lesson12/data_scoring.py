import pandas as pd
from sklearn.linear_model import LinearRegression
import os

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)

# This line of code generates
# ['1', 'x0', 'x0^2', 'x0^3', 'x0^4']
# for all x values.
X_poly   = poly_reg.fit_transform([[4]])
print(X_poly)

def scoreData(xarray, poly_reg):
    # ['1', 'x0', 'x0^2', 'x0^3', 'x0^4']
    # The polynomial tranform is an array the variable raised to
    # different powers.
    print("\nFeature Names: ")
    print(poly_reg.get_feature_names_out())

    # Show the polynomial features.
    x = poly_reg.fit_transform(xarray)[0]

    # Intercept from linear regression.
    b = 135486.37366816

    # Coefficients from linear regression.
    b1 = -143427.12888394
    b2 = 68157.93948411
    b3 = -11608.84599445
    b4 = 709.02332493

    # Multiply linear regression coefficients by polynomial features.
    predict = b * x[0] + b1 * x[1] + b2 * x[2] + b3 * x[3] + b4 * x[4]
    print("\n***Prediction: " + str(predict))

scoreData([[3]], poly_reg)