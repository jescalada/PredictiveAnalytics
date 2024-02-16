import pandas as pd
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch import optim
from skorch import NeuralNetClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# This class could be any name.
# nn.Module is needed to enable grid searching of parameters
# with skorch later.


class MyNeuralNet(nn.Module):
    # Define network objects.
    # Defaults are set for number of neurons and the
    # dropout rate.
    def __init__(self, num_neurons=10, dropout=0.1):
        super(MyNeuralNet, self).__init__()
        # 1st hidden layer.
        # nn. Linear(n,m) is a module that creates single layer
        # feed forward network with n inputs and m output.
        self.dense0 = nn.Linear(28, num_neurons)
        self.activationFunc = nn.ReLU()

        # Drop samples to help prevent overfitting.
        self.dropout = nn.Dropout(dropout)

        # 2nd hidden layer.
        self.dense1 = nn.Linear(num_neurons, num_neurons)

        # Output layer.
        self.output = nn.Linear(num_neurons, 4)

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


from skorch.callbacks import EpochScoring


def build_model(X_train, y_train):
    num_neurons = 25  # hidden layers
    net = NeuralNetClassifier(MyNeuralNet(num_neurons), max_epochs=200,
                              lr=0.001, batch_size=100, optimizer=optim.RMSprop,
                              callbacks=[EpochScoring(scoring='accuracy',
                                                      name='juan_acc', on_train=True)])
    # Pipeline execution
    model = net.fit(X_train, y_train)
    return model, net


def evaluate_model(model, X_test, y_test):
    print(model)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)


def get_customer_segmentation_data():
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    PATH = f"{ROOT_PATH}\\..\\datasets\\"

    df = pd.read_csv(PATH + 'CustomerSegmentation.csv')
    df = pd.get_dummies(df, columns=[
        'Gender', 'Ever_Married',
        'Graduated', 'Profession', 'Spending_Score', 'Var_1'])
    df['Segmentation'] = df['Segmentation'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3})
    print(df['Segmentation'].value_counts())
    X = df.copy()
    del X['Segmentation']
    y = df['Segmentation']
    return X, y


X, y = get_customer_segmentation_data()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.int64)
y_test = torch.tensor(y_test.values, dtype=torch.int64)

model, net = build_model(X_train, y_train)

evaluate_model(model, X_test, y_test)


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


draw_loss_plot(net)
draw_accuracy_plot(net)
