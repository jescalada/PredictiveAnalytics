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
accuracy = []


def build_model(batch_size):
    NUM_LAYERS = 7
    # define the keras model
    model = Sequential()
    model.add(Dense(230, input_dim=8, activation='relu', kernel_initializer='he_normal'))
    for i in range(0, NUM_LAYERS - 1):
        model.add(Dense(230, activation='relu', kernel_initializer='he_normal'))

    model.add(Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.0005, momentum=0.9, name="SGD",
    )

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # fit the keras model on the dataset
    history = model.fit(X, y, epochs=500, batch_size=batch_size,
                        validation_data=(X_test, y_test))
    # evaluate the keras model

    # Evaluate the model.
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % acc)
    accuracy.append(acc)
    print(accuracy)
    return history


def show_loss(history, batch_size):
    # Get training and test loss histories
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history for training data.
    actual_label = str(batch_size) + " batch"
    plt.subplot(1, 2, 1)
    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, label=actual_label)
    plt.legend()


def show_accuracy(history, batch_size):
    # Get training and test loss histories
    training_loss = history.history['accuracy']
    validation_loss = history.history['val_accuracy']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    plt.subplot(1, 2, 2)

    actual_label = str(batch_size) + " batch"
    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, label=actual_label)
    plt.legend()


plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
batch_sizes = [32, len(y_train)]

for i in range(0, len(batch_sizes)):
    history = build_model(batch_sizes[i])
    show_loss(history, batch_sizes[i])
    show_accuracy(history, batch_sizes[i])

plt.show()
