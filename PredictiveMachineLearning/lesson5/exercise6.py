from sklearn.datasets import make_blobs
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Generate the data.
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def prepare_data():
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    PATH = f"{ROOT_PATH}\\..\\datasets\\"

    # load the dataset
    df = pd.read_csv(PATH + 'diabetes.csv', sep=',')

    # split into input (X) and output (y) variables
    X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
            'DiabetesPedigreeFunction', 'Age']]
    y = df[['Outcome']]

    # Split into train and test data sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    return X_train, X_test, y_train, y_test


# Build the base model.
def get_base_model(train_x, train_y):
    # define the keras model
    model = Sequential()
    model.add(Dense(230, input_dim=8, activation='relu',
                    kernel_initializer='he_normal'))

    model.add(Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.0005, momentum=0.9, name="SGD",
    )

    # Compile the keras model.
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    # Fit the keras model on the dataset.
    model.fit(train_x, train_y, epochs=200, batch_size=10)

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
    model.add(Dense(230, activation='relu', kernel_initializer='he_uniform'))

    # Add the output layer back.
    model.add(output_layer)

    # fit model
    model.fit(train_x, train_y, epochs=300, verbose=1)
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

columns = ['# layers', 'train_acc', 'test_acc', 'train_loss', 'test_loss']

df = pd.DataFrame(columns=columns)
for i in range(0, len(stats)):
    df = df._append(stats[i], ignore_index=True)
print(df)
