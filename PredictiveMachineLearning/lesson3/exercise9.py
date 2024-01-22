from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Prep data.
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "winequality.csv"
dataset  = pd.read_csv(PATH + CSV_DATA)
X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
             'pH', 'sulphates','alcohol']].values
y = dataset['quality']


def get_unfit_models():
    models = list()
    models.append(ElasticNet())
    models.append(SVR(gamma='scale'))
    models.append(DecisionTreeRegressor())
    models.append(AdaBoostRegressor())
    models.append(RandomForestRegressor(n_estimators=200))
    models.append(ExtraTreesRegressor(n_estimators=200))
    return models


def evaluate_model(y_test, predictions, model):
    mse = mean_squared_error(y_test, predictions)
    rmse = round(np.sqrt(mse), 3)
    print(" RMSE:" + str(rmse) + " " + model.__class__.__name__)


def fit_base_models(X_train, y_train, X_test, models):
    dfPredictions = pd.DataFrame()

    # Fit base model and store its predictions in dataframe.
    for i in range(0, len(models)):
        models[i].fit(X_train, y_train)
        predictions = models[i].predict(X_test)
        col_name = str(i)
        # Add base model predictions to column of data frame.
        dfPredictions[col_name] = predictions
    return dfPredictions, models


def fit_stacked_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


# Split data into train, test and validation sets.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

# Get base models.
unfit_models = get_unfit_models()

# Fit base and stacked models.
df_predictions, models = fit_base_models(X_train, y_train, X_val, unfit_models)
stacked_model = fit_stacked_model(df_predictions, y_val)

# Evaluate base models with validation data.
print("\n** Evaluate Base Models **")
df_validation_predictions = pd.DataFrame()
for i in range(0, len(models)):
    predictions = models[i].predict(X_test)
    col_name = str(i)
    df_validation_predictions[col_name] = predictions
    evaluate_model(y_test, predictions, models[i])

# Evaluate stacked model with validation data.
stacked_predictions = stacked_model.predict(df_validation_predictions)
print("\n** Evaluate Stacked Model **")
evaluate_model(y_test, stacked_predictions, stacked_model)

# Show the dataframe used to fit the stacked model
# Make it reader-friendly by adding labels to the columns.
df_predictions.columns = ['ElasticNet', 'SVR', 'DecisionTree', 'AdaBoost', 'RandomForest', 'ExtraTrees']
print(df_predictions.head())
