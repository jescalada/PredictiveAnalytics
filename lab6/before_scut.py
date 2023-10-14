import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, \
    precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE = 'healthcare-dataset-stroke-data.csv'
data = pd.read_csv(PATH + FILE)


def show_y_plots(y_train, y_test, title):
    """
    Shows the distribution of the target variable in training and test data.

    :param y_train: Distribution of target variable in training data.
    :param y_test: Distribution of target variable in test data.
    :param title: Title of the plot.
    """
    print("\n ***" + title)
    plt.subplots(1, 2)

    plt.subplot(1, 2, 1)
    plt.hist(y_train)
    plt.title("Train Y: " + title)

    plt.subplot(1, 2, 2)
    plt.hist(y_test)
    plt.title("Test Y: " + title)
    plt.show()


def evaluate_model(X_test, y_test, y_train, model, title):
    """
    Evaluates the model by showing the confusion matrix, precision, recall, and accuracy.

    :param X_test: Test data.
    :param y_test: Test target variable.
    :param y_train: Training target variable.
    :param model: Model to evaluate.
    :param title: Title of the plot.
    """
    show_y_plots(y_train, y_test, title)

    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    print(cm)
    precision = precision_score(y_test, preds, average='binary')
    print("Precision: " + str(precision))

    recall = recall_score(y_test, preds, average='binary')
    print("Recall:    " + str(recall))

    accuracy = accuracy_score(y_test, preds)
    print("Accuracy:    " + str(accuracy))


# Inspect data.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(data.head())
print(data.describe())

# Impute missing bmi values with average BMI value.
averageBMI = np.mean(data['bmi'])
data['bmi'] = data['bmi'].replace(np.nan, averageBMI)
print(data.describe())


def get_train_and_test_data(data):
    """
    Splits the data into training and test sets.
    :param data: Data to split.
    :return: Four arrays: X_train, X_test, y_train, y_test
    """
    X = data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]
    y = data['stroke']

    # Split the data into train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = get_train_and_test_data(data)

# Build logistic regressor and evaluate model.
clf = LogisticRegression(solver='newton-cg', max_iter=1000)
clf.fit(X_train, y_train)
evaluate_model(X_test, y_test, y_train, clf, "Before SCUT")
