import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os
import numpy as np

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = ROOT_PATH + "\\..\\datasets\\"
CSV_DATA = "usCityData.csv"
df = pd.read_csv(PATH + CSV_DATA)
X = df[['lstat']]
y = df[['medv']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10,
                                                    random_state=42,
                                                    shuffle=True)
# Draw scatter plot of the test set.
plt.scatter(X_test, y_test)

# Train the model.
degree = 5
poly_reg = PolynomialFeatures(degree=degree)
transformed_x = poly_reg.fit_transform(X_train)
linReg = LinearRegression()

# Generate linear regression model with transformed x.
linReg.fit(transformed_x, y_train)

# Sort the test data.
sortedTestDf = X_test[['lstat']]
sortedTestDf['medv'] = y_test[['medv']]
sortedTestDf = sortedTestDf.sort_values(['lstat'], ascending=[True])

# Generate predictions with sorted test data.
transformed_x_test = poly_reg.fit_transform(sortedTestDf[['lstat']])
predictions = linReg.predict(transformed_x_test)
print(predictions)

# Calculate the RMSE using the predictions.
rmse = np.sqrt(np.sum((y_test - predictions) ** 2) / len(predictions))
print(f"RMSE: {rmse}")

# Visualize the prediction line against the scatter plot.
plt.plot(sortedTestDf[['lstat']], predictions, label="degree %d" % degree
                                                     + '; $R^2$: %.2f' % linReg.score(transformed_x_test,
                                                                                      sortedTestDf['medv']))

# Show the coefficients and intercept.
print(f"Coefficients: {linReg.coef_}")
print(f"Intercept: {linReg.intercept_}")

def scoreData(xarray, poly_reg):
    # The polynomial tranform is an array the variable raised to
    # different powers.
    print("\nFeature Names: ")
    print(poly_reg.get_feature_names_out())

    # Show the polynomial features.
    x = poly_reg.fit_transform(xarray)[0]

    # Intercept from linear regression.
    b = 66.07760144

    # Coefficients from linear regression.
    b1 = -1.13110672e+01
    b2 = 1.18244444e+00
    b3 = -6.30055149e-02
    b4 = 1.58295194e-03
    b5 = -1.48281497e-05

    # Multiply linear regression coefficients by polynomial features.
    predict = b * x[0] + b1 * x[1] + b2 * x[2] + b3 * x[3] + b4 * x[4] + b5 * x[5]
    print("\n***Prediction: " + str(predict))


plt.legend(loc='upper right')
plt.xlabel("Test LSTAT Data")
plt.ylabel("Predicted Price")
plt.title("Variance Explained with Varying Polynomial")
plt.show()

# Test the scoreData function using lstat = 2.88
scoreData([[2.88]], poly_reg)
