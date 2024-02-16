from sklearn.datasets import make_classification
from torch import optim
from skorch import NeuralNetClassifier
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# This class could be any name.
# nn.Module is needed to enable grid searching of parameters
# with skorch later.
class MyNeuralNet(nn.Module):
    # Define network objects.
    # Defaults are set for number of neurons and the
    # dropout rate.
    def __init__(self, num_neurons=50, dropout=0.1):
        super(MyNeuralNet, self).__init__()
        # 1st hidden layer.
        # nn. Linear(n,m) is a module that creates single layer
        # feed forward network with n inputs and m output.
        self.dense0 = nn.Linear(2, num_neurons)
        print("Dense layer type:")
        print(self.dense0.weight.dtype)

        self.activationFunc = nn.ReLU()

        # Drop samples to help prevent overfitting.
        self.dropout = nn.Dropout(dropout)

        # 2nd hidden layer.
        self.dense1 = nn.Linear(num_neurons, num_neurons)

        # Output layer.
        self.output = nn.Linear(num_neurons, 2)

        # Softmax activation function allows for multiclass predictions.
        # In this case the prediction is binary.
        self.softmax = nn.Softmax(dim=-1)

    # Move data through the different network objects.
    def forward(self, x):
        # Pass data from 1st hidden layer to activation function
        # before sending to next layer.
        X = self.activationFunc(self.dense0(x))
        X = self.dropout(X)
        X = self.activationFunc(self.dense1(X))
        X = self.softmax(self.output(X))
        return X


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def build_model(x, y):
    # Through a grid search, the optimal hyperparameters are found
    # A pipeline is used in order to scale and train the neural net
    # The grid search module from scikit-learn wraps the pipeline
    x.to(torch.float32)
    # The Neural Net is instantiated, none hyperparameter is provided
    nn = NeuralNetClassifier(MyNeuralNet, verbose=1, train_split=False)
    # The pipeline is instantiated, it wraps scaling and training phase
    pipeline = Pipeline([('nn', nn)])

    # The parameters for the grid search are defined
    # Must use prefix "nn__" when setting hyperparamters for the training phase
    # Must use prefix "nn__module__" when setting hyperparameters for the Neural Net
    params = {
        'nn__max_epochs': [10, 20],
        'nn__lr': [0.1, 0.01],
        'nn__module__num_neurons': [5, 10],
        'nn__module__dropout': [0.1, 0.5],
        'nn__optimizer': [optim.Adam, optim.SGD, optim.RMSprop]}

    # The grid search module is instantiated
    gs = GridSearchCV(pipeline, params, refit=True, cv=3,
                      scoring='balanced_accuracy', verbose=1)
    return gs.fit(x, y), nn


def evaluate_model(model, X_test, y_test):
    print(model)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)


# Setup data.
import pandas as pd
import numpy as np
import torch
import os

# Load the data.
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE = 'fluDiagnosis.csv'

df = pd.read_csv(PATH + FILE)

y = np.array(df['Diagnosed'])
X = df.copy()

del X['Diagnosed']
X = X

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# define standard scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# transform data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)  # whole number needed
y_test = torch.tensor(y_test, dtype=torch.long)  # for classification.

# Build the model.
model, net = build_model(X_train, y_train)

from matplotlib import pyplot as plt

def draw_loss_plot(net):
    plt.plot(net.history[:, 'train_loss'], color='blue', label='train')
    plt.plot(net.history[:, 'valid_loss'], color='orange', label='val')
    plt.legend()
    plt.show()


def draw_accuracy_plot(net):
    plt.plot(net.history[:, 'juan_acc'], color='blue', label='train')
    plt.plot(net.history[:, 'valid_acc'], color='orange', label='val')
    plt.legend()
    plt.show()


print("Best parameters:")
print(model.best_params_)

print("Juan's network model for college admissions.")
# Evaluate the model.
evaluate_model(model.best_estimator_, X_test, y_test)

draw_loss_plot(net)
draw_accuracy_plot(net)
