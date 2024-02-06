from sklearn.datasets import make_blobs
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
import matplotlib.pyplot as plt


# Generate the data.
def prepare_data():
    X, y = make_blobs(n_samples=1000, centers=3,
                      n_features=2, cluster_std=2, random_state=2)
    y = to_categorical(y)
    n_train = 500
    train_x, test_x = X[:n_train, :], X[n_train:, :]
    train_y, test_y = y[:n_train], y[n_train:]
    return train_x, test_x, train_y, test_y


# Build the base model.
def get_base_model(train_x, train_y):
    # Define the model.
    model = Sequential()
    model.add(Dense(10, input_dim=2, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(3, activation='softmax'))

    # Compile the model.
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

    # Fit the model.
    model.fit(train_x, train_y, epochs=100, verbose=0)
    return model


stats = []


# Evaluate the model.
def evaluate_model(num_layers, model, train_x, test_x, train_y, test_y):
    train_loss, train_acc = model.evaluate(train_x, train_y, verbose=1)
    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=1)
    stats.append({'# layers': num_layers, 'train_acc': train_acc, 'test_acc': test_acc,
                  'train_loss': train_loss, 'test_loss': test_loss})


# Add one new layer and re-train only the new layer.
def add_layer(model, train_x, train_y):
    # Store the output layer.
    output_layer = model.layers[-1]

    # Remove the output layer.
    model.pop()

    # Mark all remaining layers as non-trainable.
    for layer in model.layers:
        layer.trainable = False

    # Add a new hidden layer.
    model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))

    # Add the output layer back.
    model.add(output_layer)

    # fit model
    model.fit(train_x, train_y, epochs=100, verbose=1)
    return model


# Get the data and build the base model.
train_x, test_x, train_y, test_y = prepare_data()
model = get_base_model(train_x, train_y)

# Evaluate the base model
scores = dict()
evaluate_model(-1, model, train_x, test_x, train_y, test_y)

# add layers and evaluate the updated model
n_layers = 10
for i in range(n_layers):
    model = add_layer(model, train_x, train_y)
    evaluate_model(i, model, train_x, test_x, train_y, test_y)

import pandas as pd

columns = ['# layers', 'train_acc', 'test_acc', 'train_loss', 'test_loss']

df = pd.DataFrame(columns=columns)
for i in range(0, len(stats)):
    df = df._append(stats[i], ignore_index=True)
print(df)
