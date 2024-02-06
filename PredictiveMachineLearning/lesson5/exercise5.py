# mlp with unscaled data for the regression problem
from sklearn.datasets import make_regression
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt

# Generate the regression dataset.
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)

plt.hist(y)
plt.title("Unscaled Input")
plt.show()

# Split into train and test.
n_train = 500
train_x, test_x = X[:n_train, :], X[n_train:, :]
train_y, test_y = y[:n_train], y[n_train:]

clip_results = []


def build_model(clip_value):
    # Define the model.
    model = Sequential()
    model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='linear'))

    # Compile the model.
    opt = SGD(lr=0.01, momentum=0.9, clipvalue=clip_value)
    model.compile(loss='mean_squared_error', optimizer=opt)

    # Fit the model.
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=100, verbose=1)

    # Evaluate the model.
    train_mse = model.evaluate(train_x, train_y, verbose=0)
    test_mse = model.evaluate(test_x, test_y, verbose=0)
    print('Train MSE: %.3f, Test MSE: %.3f' % (train_mse, test_mse))
    clip_results.append({'train mse': train_mse, 'test mse': test_mse,
                         'clip value': clip_value})
    # Plot losses during training.
    plt.title('Losses')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


clip_values = [0.1, 0.5, 1.0, 2.5, 5.0]
for i in range(0, len(clip_values)):
    build_model(clip_values[i])

for clip_result in clip_results:
    print(clip_result)
