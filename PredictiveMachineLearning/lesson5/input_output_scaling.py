from sklearn.datasets import make_regression
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from matplotlib import pyplot

# Generate regression set.
X, y = make_regression(n_samples=1000, n_features=20,
                       noise=0.1, random_state=1)

# Split data into train and test.
n_train = 500
train_x, test_x = X[:n_train, :], X[n_train:, :]
train_y, test_y = y[:n_train], y[n_train:]

norm_size_evaluations = []

# Define the model.
model = Sequential()
model.add(Dense(25, input_dim=20, activation='relu',
                kernel_initializer='he_uniform'))
model.add(Dense(1, activation='linear'))

# Compile the model.
model.compile(loss='mean_squared_error',
              optimizer=SGD(lr=0.01, momentum=0.9))

from sklearn.preprocessing import StandardScaler

# reshape 1d arrays to 2d arrays
train_y = train_y.reshape(len(train_y), 1)
test_y = test_y.reshape(len(train_y), 1)

# Scale y
scaler = StandardScaler()
scaler.fit(train_y)
train_y = scaler.transform(train_y)
test_y = scaler.transform(test_y)

# Scale x
x_scaler = StandardScaler()
x_scaler.fit(train_x)
train_x = x_scaler.transform(train_x)
test_x = x_scaler.transform(test_x)

# Fit the model.
history = model.fit(train_x, train_y,
                    validation_data=(test_x, test_y),
                    epochs=200, verbose=1)

# Evaluate the model.
train_mse = model.evaluate(train_x, train_y, verbose=0)
test_mse = model.evaluate(test_x, test_y, verbose=0)
print('Train loss: %.3f, Test loss: %.3f' % (train_mse, test_mse))

norm_size_evaluations.append({'train mse': train_mse,
                            'test mse': test_mse,
                            'size': 1})

# Plot the loss during training.
pyplot.title('Mean Squared Error - norm size: ')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
