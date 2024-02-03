import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import warnings
import os

warnings.filterwarnings(action='once')

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "housing.data"
df = pd.read_csv(PATH + CSV_DATA, header=None)

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

print(df.head())
print(df.tail())
print(df.describe())

dataset = df.values

# split into input (X) and output (Y) variables
X = dataset[:, 0:13]
y = dataset[:, 13]

train_x, temp_X, train_y, temp_y = train_test_split(X, y, train_size=0.7)
valX, testX, valY, testY = train_test_split(temp_X, temp_y, train_size=0.5)

# Scale X and Y.
sc_x = StandardScaler()
scaler_x = sc_x.fit(train_x)
train_x_scaled = scaler_x.transform(train_x)
val_x_scaled = scaler_x.transform(valX)
test_x_scaled = scaler_x.transform(testX)

sc_y = StandardScaler()
train_y_scaled = sc_y.fit_transform(np.array(train_y).reshape(-1, 1))
test_y_scaled = sc_y.transform(np.array(testY).reshape(-1, 1))
val_y_scaled = sc_y.transform(np.array(valY).reshape(-1, 1))

# Build basic multilayer perceptron.
model1 = MLPRegressor(
    # 3 hidden layers with 150 neurons, 100, and 50.
    hidden_layer_sizes=(150, 100, 50),
    max_iter=50,  # epochs
    activation='relu',
    solver='adam',  # optimizer
    verbose=1)
model1.fit(train_x_scaled, train_y_scaled)


def show_losses(model):
    plt.plot(model.loss_curve_)
    plt.title("Loss Curve")
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()


def evaluate_model(model, test_x_scaled, test_y_scaled, sc_y):
    scaled_predictions = model.predict(test_x_scaled)
    y_pred = sc_y.inverse_transform(
        np.array(scaled_predictions).reshape(-1, 1))
    mse = metrics.mean_squared_error(test_y_scaled, y_pred)
    rmse = np.sqrt(mse)
    print("RMSE: " + str(rmse))


evaluate_model(model1, val_x_scaled, val_y_scaled, sc_y)

param_grid = {
    'hidden_layer_sizes': [(150, 100, 50), (120, 80, 40), (100, 50, 30)],
    'max_iter': [50, 100],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2]
}

model2 = GridSearchCV(estimator=model1, param_grid=param_grid,
                      n_jobs=-1, cv=4, verbose=1,
                      return_train_score=False)

model2.fit(train_x_scaled, train_y_scaled)
print("Best parameters: ")
print(model2.best_params_)
y_pred = model2.predict(test_x_scaled)

evaluate_model(model2, val_x_scaled, val_y_scaled, sc_y)

best_params = {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (150, 100, 50), 'learning_rate': 'adaptive', 'learning_rate_init': 0.2, 'max_iter': 100, 'solver': 'sgd'}

model3 = MLPRegressor(**best_params)
model3.fit(train_x_scaled, train_y_scaled)
evaluate_model(model3, val_x_scaled, val_y_scaled, sc_y)
