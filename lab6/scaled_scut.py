import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, \
    precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from crucio import SCUT

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
    preds = model.predict(X_test)
    print("\n *** " + title)

    cm = confusion_matrix(y_test, preds)
    print(cm)
    precision = precision_score(y_test, preds, average='binary')
    print("Precision: " + str(precision))

    recall = recall_score(y_test, preds, average='binary')
    print("Recall:    " + str(recall))

    accuracy = accuracy_score(y_test, preds)
    print("Accuracy:    " + str(accuracy))

    if precision + recall == 0.0:
        f1 = 0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    print("F1:    " + str(f1))


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

# Boost representation of minority in training data with SCUT.
dfTrain = X_train.copy()
dfTrain['stroke'] = y_train
scut = SCUT()
df_scut = scut.balance(dfTrain, 'stroke')

# Adjust y_train and X_train with better represented minority.
y_train = df_scut['stroke']
X_train = df_scut
del df_scut['stroke']

# Perform logistic regression with SCUT-treated train data and
# evaluate with untouched test data.
clf = LogisticRegression(solver='newton-cg', max_iter=1000)
clf.fit(X_train, y_train)
evaluate_model(X_test, y_test, y_train, clf, "After SCUT")

# Perform logistic regression with SCUT-treated train data and MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()

X_train_scaled_minmax = minmax_scaler.fit_transform(X_train)
X_test_scaled_minmax = minmax_scaler.transform(X_test)

clf = LogisticRegression(solver='newton-cg', max_iter=1000)
clf.fit(X_train_scaled_minmax, y_train)
evaluate_model(X_test_scaled_minmax, y_test, y_train, clf, "After SCUT and MinMaxScaler")

# Perform logistic regression with SCUT-treated train data and StandardScaler

from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()

X_train_scaled_standard = standard_scaler.fit_transform(X_train)
X_test_scaled_standard = standard_scaler.transform(X_test)

clf = LogisticRegression(solver='newton-cg', max_iter=1000)
clf.fit(X_train_scaled_standard, y_train)
evaluate_model(X_test_scaled_standard, y_test, y_train, clf, "After SCUT and StandardScaler")

# Perform logistic regression with SCUT-treated train data and RobustScaler

from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()

X_train_scaled_robust = robust_scaler.fit_transform(X_train)
X_test_scaled_robust = robust_scaler.transform(X_test)

clf = LogisticRegression(solver='newton-cg', max_iter=1000)
clf.fit(X_train_scaled_robust, y_train)
evaluate_model(X_test_scaled_robust, y_test, y_train, clf, "After SCUT and RobustScaler")