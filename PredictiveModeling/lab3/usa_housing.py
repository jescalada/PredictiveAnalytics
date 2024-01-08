import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_predicted_vs_actual(y_test, predictions, model_name):
    # Plot predicted vs actual
    plt.scatter(y_test, predictions)
    plt.legend()
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f'Predicted (Y) vs. Actual (X): {model_name}')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.show()


def plot_residuals_vs_actual(y_test, predictions, model_name):
    # Plot residuals vs actual
    residuals = y_test - predictions
    plt.scatter(y_test, residuals, label='Residuals vs Actual')
    plt.xlabel("Actual")
    plt.ylabel("Residual")
    plt.title(f'Error Residuals (Y) vs. Actual (X): {model_name}')
    plt.show()


def perform_analysis(X, y, test_size=0.2, model_name=""):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model

    print(model.summary())
    print(f'{model_name} RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    plot_predicted_vs_actual(y_test, predictions, model_name)
    BINS = 50
    drawValidationPlots(model_name, BINS, y_test, predictions)

    return predictions, model



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


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "USA_Housing.csv"

dataset = pd.read_csv(PATH + CSV_DATA,
                      skiprows=1,  # Don't include header row as part of data.
                      encoding="ISO-8859-1", sep=',',
                      names=('Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                             'Avg. Area Number of Bedrooms', "Area Population", 'Price', "Address"))
# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(dataset.head(3))

# Show statistical summary of numerical columns
print(dataset.describe())

# Show correlation heatmap for numerical columns (not including Address)
X = dataset[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
             "Area Population", 'Price']]
corr = X.corr()

# plot the heatmap, enlarge to fit the labels
heatmap = sns.heatmap(corr,
                      xticklabels=corr.columns,
                      yticklabels=corr.columns,
                      annot=True,
                      annot_kws={"size": 4},
                      linewidths=0.1,
                      vmax=1,
                      cmap="YlGnBu",
                      square=True,
                      cbar_kws={"shrink": .4})

plt.figure(figsize=(20, 10))
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=8)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, horizontalalignment='right', fontsize=8)
plt.show()

# MODEL 1
# Adding an intercept to prevent overfitting
X_no_price = dataset[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                      'Avg. Area Number of Bedrooms', "Area Population"]]
X = sm.add_constant(X_no_price)
y = dataset['Price'].values

perform_analysis(X, y, test_size=0.1, model_name="All Features Model")

# MODEL 2
# Remove Avg. Area Number of Bedrooms
X_no_bedrooms = dataset[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Area Population']]
X = sm.add_constant(X_no_bedrooms)
y = dataset['Price'].values

perform_analysis(X, y, test_size=0.1, model_name="No Bedrooms Model")


# MODEL 3
# Remove Avg. Area Number of Rooms
X_no_rooms = dataset[['Avg. Area Income', 'Avg. Area House Age', 'Area Population']]
X = sm.add_constant(X_no_rooms)
y = dataset['Price'].values

perform_analysis(X, y, test_size=0.1, model_name="No Rooms Model")

# Get three random samples from the dataset
print(dataset.sample(n=3))