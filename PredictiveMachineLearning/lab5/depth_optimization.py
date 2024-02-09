from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import os

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


def build_model(num_layers):
    # define the keras model
    model = Sequential()
    model.add(Dense(230, input_dim=8, activation='relu',
                    kernel_initializer='he_normal'))

    for i in range(0, num_layers - 1):
        model.add(Dense(230, activation='relu',
                        kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.0005, momentum=0.9, name="SGD",
    )

    # Compile the keras model.
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    # Fit the keras model on the dataset.
    history = model.fit(X, y, epochs=200, batch_size=10,
                        validation_data=(X_test, y_test))

    # Evaluate the model.
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % acc)
    return history


def show_loss(history, num_nodes):
    # Get training and test loss histories
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history for training data.
    actual_label = str(num_nodes) + " layers"
    plt.subplot(1, 2, 1)

    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, label=actual_label)
    plt.legend()


def show_accuracy(history, num_nodes):
    # Get training and test loss histories
    training_loss = history.history['accuracy']
    validation_loss = history.history['val_accuracy']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    plt.subplot(1, 2, 2)

    actual_label = str(num_nodes) + " layers"
    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, label=actual_label)
    plt.legend()


num_layers = [5, 6, 7, 8]
plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

for i in range(0, len(num_layers)):
    history = build_model(num_layers[i])
    show_loss(history, num_layers[i])
    show_accuracy(history, num_layers[i])

plt.show()
