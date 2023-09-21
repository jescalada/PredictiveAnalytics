import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "winequality.csv"
dataset = pd.read_csv(DATASET_DIR + CSV_DATA,
                      skiprows=1,       # Don't include header row as part of data.
                      encoding="ISO-8859-1", sep=',',
                      names=('fixed acidity', 'volatile acidity', 'citric acid',
                             'residual sugar', 'chlorides', 'free sulfur dioxide',
                             'total sulfur dioxide', 'density', 'pH', 'sulphates',
                             'alcohol', 'quality'))
# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)
print(dataset.head())
print(dataset.describe())
X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
             'pH', 'sulphates','alcohol']].values

# Adding an intercept (required). Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X = sm.add_constant(X)

y = dataset['quality'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)  # make the predictions by the model

print(model.summary())

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

def plotPredictionVsActual(plt, title, y_test, predictions):
    plt.scatter(y_test, predictions)
    plt.legend()
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title('Predicted (Y) vs. Actual (X): ' + title)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')


def plotResidualsVsActual(plt, title, y_test, predictions):
    residuals = y_test - predictions
    plt.scatter(y_test, residuals, label='Residuals vs Actual')
    plt.xlabel("Actual")
    plt.ylabel("Residual")
    plt.title('Error Residuals (Y) vs. Actual (X): ' + title)
    plt.plot([y_test.min(), y_test.max()], [0, 0], 'k--')


def plotResidualHistogram(plt, title, y_test, predictions, bins):
    residuals = y_test - predictions
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.hist(residuals, label='Residuals vs Actual', bins=bins)
    plt.title('Error Residual Frequency: ' + title)
    plt.plot()


def drawValidationPlots(title, bins, y_test, predictions):
    # Define number of rows and columns for graph display.
    plt.subplots(nrows=1, ncols=3, figsize=(12, 5))

    plt.subplot(1, 3, 1)  # Specfy total rows, columns and image #
    plotPredictionVsActual(plt, title, y_test, predictions)

    plt.subplot(1, 3, 2)  # Specfy total rows, columns and image #
    plotResidualsVsActual(plt, title, y_test, predictions)

    plt.subplot(1, 3, 3)  # Specfy total rows, columns and image #
    plotResidualHistogram(plt, title, y_test, predictions, bins)
    plt.show()


BINS = 8
TITLE = "Wine Quality"
drawValidationPlots(TITLE, BINS, y_test, predictions)
