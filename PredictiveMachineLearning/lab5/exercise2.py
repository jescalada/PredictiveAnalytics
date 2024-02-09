import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import tensorflow as tf
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"

# load the dataset
df = pd.read_csv(PATH + 'fluDiagnosis.csv')
# split into input (X) and output (y) variables
print(df)

X = df[['A', 'B']]
y = df[['Diagnosed']]
# Split into train and test data sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def build_model(num_nodes, learning_rate, momentum):
    # define the keras model
    model = Sequential()
    model.add(Dense(num_nodes, input_dim=2, activation='relu',
                    kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=momentum, name="SGD",
    )

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    # fit the keras model on the dataset
    history = model.fit(X, y, epochs=80, batch_size=32, validation_data=(X_test,
                                                                         y_test))
    # evaluate the keras model

    # Evaluate the model.
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: ' + str(acc) + ' Num nodes: ' + str(num_nodes))
    return history


def show_loss(history, num_nodes, learning_rate, momentum):
    # Get training and test loss histories
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history for training data.
    actual_label = str(num_nodes) + " nodes" + " lr: " + str(learning_rate) + " momentum: " + str(momentum)
    plt.subplot(1, 2, 1)
    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, label=actual_label)
    plt.legend()


def show_accuracy(history, num_nodes, learning_rate, momentum):
    # Get training and test loss histories
    training_loss = history.history['accuracy']
    validation_loss = history.history['val_accuracy']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    plt.subplot(1, 2, 2)

    actual_label = str(num_nodes) + " nodes" + " lr: " + str(learning_rate) + " momentum: " + str(momentum)
    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, label=actual_label)
    plt.legend()


node_counts = [50, 100, 200]
learning_rates = [0.0001]
momentums = [0.9, 0.99]

plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

for node_count in node_counts:
    for learning_rate in learning_rates:
        for momentum in momentums:
            history = build_model(node_count, learning_rate, momentum)
            show_loss(history, node_count, learning_rate, momentum)
            show_accuracy(history, node_count, learning_rate, momentum)

plt.show()
