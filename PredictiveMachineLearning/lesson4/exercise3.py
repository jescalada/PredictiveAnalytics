import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
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

# Convert DataFrame columns to vertical columns so they can be used by the NN.
dataset = df.values
X = dataset[:, 0:13]  # Columns 0 to 12
y = dataset[:, 13]  # Columns 13
ROW_DIM = 0
COL_DIM = 1

x_array_reshaped = X.reshape(X.shape[ROW_DIM], X.shape[COL_DIM])
y_array_reshaped = y.reshape(y.shape[ROW_DIM], 1)

# Split the data.
X_train, X_temp, y_train, y_temp = train_test_split(x_array_reshaped,
                                                    y_array_reshaped, test_size=0.3, random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_temp,
                                                y_temp, test_size=0.5, random_state=0)


# Define the model.
def create_model(learning_rate=0.01):
    model = Sequential()

    from keras.optimizers import Adam
    optimizer = Adam(lr=learning_rate)

    model.add(Dense(13, input_dim=13, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


model = create_model()

# Build the model.
model = create_model(learning_rate=0.005)
history = model.fit(X_train, y_train, epochs=100,
                    batch_size=10, verbose=1,
                    validation_data=(X_val, y_val))

# Evaluate the model.
predictions = model.predict(X_test)
mse = metrics.mean_squared_error(y_test, predictions)
print("Neural network MSE: " + str(mse))
print("Neural network RMSE: " + str(np.sqrt(mse)))
