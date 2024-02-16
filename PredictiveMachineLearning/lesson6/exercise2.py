from sklearn.datasets import make_classification
from torch import optim
from skorch import NeuralNetClassifier
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import pandas as pd
import torch


# This class could be any name.
# nn.Module is needed to enable grid searching of parameters
# with skorch later.
class MyNeuralNet(nn.Module):
    # Define network objects.
    # Defaults are set for number of neurons and the
    # dropout rate.
    def __init__(self, num_neurons=10, dropout=0.1, num_features=4):
        super(MyNeuralNet, self).__init__()
        # 1st hidden layer.
        # nn. Linear(n,m) is a module that creates single layer
        # feed forward network with n inputs and m output.
        self.dense0 = nn.Linear(num_features, num_neurons)
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


def build_model(x, y, num_features):
    # Trains the Neural Network with fixed hyperparameters
    # The Neural Net is initialized with fixed hyperparameters
    my_network = MyNeuralNet(num_neurons=20, dropout=0.1, num_features=num_features)
    nn = NeuralNetClassifier(my_network, max_epochs=21,
                             lr=0.01, batch_size=12,
                             optimizer=optim.RMSprop)
    model = nn.fit(x, y)
    return model


def evaluate_model(model, X_test, y_test):
    print(model)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"

df = pd.read_csv(PATH + 'bill_authentication.csv')
X = df.copy()
del X['Class']
y = df['Class']

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler

scalerX = StandardScaler()
scaledXTrain = scalerX.fit_transform(X_train)
scaledXTest = scalerX.transform(X_test)

# Columns: Variance,Skewness,Kurtosis,Entropy,Class

# # Must convert the data to PyTorch tensors
X_train_tensor = torch.tensor(scaledXTrain, dtype=torch.float32)
X_test_tensor = torch.tensor(scaledXTest, dtype=torch.float32)
y_train_tensor = torch.tensor(list(y_train), dtype=torch.long)
y_test_tensor = torch.tensor(list(y_test), dtype=torch.long)

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)
# print("X_train_tensor shape: ", X_train_tensor.shape)
# print("y_train_tensor shape: ", y_train_tensor.shape)
# print("X_test_tensor shape: ", X_test_tensor.shape)
# print("y_test_tensor shape: ", y_test_tensor.shape)
print("X: ", X)
print("y: ", y)
# Build the model.
model = build_model(X_train_tensor, y_train_tensor, num_features=4)

# Evaluate the model.
evaluate_model(model, X_test_tensor, y_test_tensor)
