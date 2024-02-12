from keras.models import Sequential
from keras.layers import Dense
from os import makedirs
from os import path
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from keras.utils import to_categorical
import pandas as pd
import numpy as np

PATH = './models/'


# fit model on dataset
def fit_model(train_x, train_y):
    # define model
    model = Sequential()
    model.add(Dense(25, input_dim=2, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(train_x, train_y, epochs=500, verbose=0)
    return model


def generate_data():
    # generate 2d classification dataset
    X, y = make_blobs(n_samples=800, centers=3,
                      n_features=2,
                      cluster_std=2, random_state=2)

    # split into train and test
    train_x, temp_x, train_y, temp_y = train_test_split(X, y, test_size=0.6)
    val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5)
    return train_x, val_x, test_x, train_y, val_y, test_y


def generate_models(train_x, train_y):
    # create directory for models
    if not path.exists(PATH):
        makedirs('./models')

    # fit and save models
    num_models = 5
    print("\nFitting models with training data.")
    for i in range(num_models):
        # fit model
        model = fit_model(train_x, train_y)
        # save model
        filename = PATH + 'model_' + str(i + 1) + '.h5'
        model.save(filename)
        print('>Saved %s' % filename)


train_x, val_x, test_x, train_y, val_y, test_y = generate_data()

# one hot encode output variable
train_y = to_categorical(train_y)
generate_models(train_x, train_y)


# load models from file
def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = PATH + 'model_' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of models
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


# trainX, valX, trainy, valY = generate_data()

# load all models
num_models = 5
models = load_all_models(num_models)
print('Loaded %d models' % len(models))

print("\nEvaluating single models with test data.")
# evaluate standalone models o821
# Model Accuracy: 0.821n test dataset
# individual ANN models are built with one-hot encoded data.
for model in models:
    one_hot_encoded_y = to_categorical(test_y)
    _, acc = model.evaluate(test_x, one_hot_encoded_y, verbose=0)
    print('Model Accuracy: %.3f' % acc)


# create stacked model input dataset as outputs from the ensemble
def get_stacked_data(models, inputX):
    stack_xdf = None
    for model in models:
        # make prediction
        yhat = model.predict(inputX, verbose=0)
        single_model_pred_df = pd.DataFrame(np.row_stack(yhat))

        # Store predictions of all models for 1 sample in each df row.
        # Here is 1st row for 5 models with predictions for 3 classes each.
        # 5 models x 3 classes = 15 columns.
        #          0             1         2   ...        12            13        14
        # 0 0.993102  1.106366e-04  0.006788   ...  0.993102  1.106366e-04  0.006788
        if stack_xdf is None:
            stack_xdf = single_model_pred_df
        else:
            num_classes = len(single_model_pred_df.keys())
            num_stack_x_cols = len(stack_xdf.keys())

            # Add new classification columns.
            for i in range(0, num_classes):
                stack_xdf[num_stack_x_cols + i] = stack_xdf[i]
    return stack_xdf


# Make predictions with the stacked model
def stacked_prediction(models, stacked_model, input_x):
    # create dataset using ensemble
    stacked_x = get_stacked_data(models, input_x)
    # make a prediction
    y_hat = stacked_model.predict(stacked_x)
    return y_hat


# fit a model based on the outputs from the ensemble models
def fit_stacked_model(models, input_x, inputy):
    # create dataset using ensemble
    stacked_x = get_stacked_data(models, input_x)
    # fit standalone model
    model = LogisticRegression()
    model.fit(stacked_x, inputy)
    return model


# fit stacked model using the ensemble
# Stacked model build with LogisticRegression.
# y for LogisticRegression is not one-hot encoded.
print("\nFitting stacked model with test data.")
stacked_model = fit_stacked_model(models, val_x, val_y)

# evaluate model on test set
print("\nEvaluating stacked model with test data.")
yhat = stacked_prediction(models, stacked_model, test_x)
acc = accuracy_score(test_y, yhat)
print('Stacked Test Accuracy: %.3f' % acc)
