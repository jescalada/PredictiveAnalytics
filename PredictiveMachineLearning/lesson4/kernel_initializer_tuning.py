import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "housing.data"
df = pd.read_csv(PATH + CSV_DATA, header=None)

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(df.head())
print(df.tail())
print(df.describe())

dataset = df.values

# split into input (X) and output (Y) variables
X = dataset[:, 0:13]
y = dataset[:, 13]

# Split the data.
X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                    y, test_size=0.3, random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_temp,
                                                y_temp, test_size=0.5, random_state=0)


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print("RMSE: " + str(rmse))
    return rmse


def show_results(network_stats):
    df_stats = pd.DataFrame.from_records(network_stats)
    df_stats = df_stats.sort_values(by=['rmse'])
    print(df_stats)


network_stats = []

kernel_initializers = ['uniform', 'he_uniform', 'glorot_uniform', 'lecun_uniform', 'normal', 'he_normal', 'glorot_normal', 'zero']

# Build model
def create_model(initializer='normal'):
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(1, kernel_initializer=initializer))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


for initializer in kernel_initializers:
    model = create_model(initializer=initializer)
    history = model.fit(X_train, y_train, epochs=100,
                        batch_size=10, verbose=1,
                        validation_data=(X_val, y_val))
    rmse = evaluate_model(model, X_test, y_test)
    network_stats.append({"rmse": rmse, "epochs": 100, "batch": 10, "initializer": initializer})
show_results(network_stats)
