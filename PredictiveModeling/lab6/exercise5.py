import os
import pandas as pd
from crucio import SCUT
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def showYPlots(y_train, y_test, title):
    """
    Shows the distribution of the target variable in training and test data.

    :param y_train: The distribution of the target variable in training data.
    :param y_test: The distribution of the target variable in test data.
    :param title: The title of the plot.
    """
    print("\n ***" + title)
    plt.subplots(1,2)

    plt.subplot(1,2 ,1)
    plt.hist(y_train)
    plt.title("Train Y: " + title)

    plt.subplot(1,2, 2)
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


def get_train_and_test_data(data):
    """
    Splits the data into training and test sets.
    :param data: Data to split.
    :return: Four arrays: X_train, X_test, y_train, y_test
    """
    X = data[['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp',
          'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]
    y = data[['TenYearCHD']]

    # Split the data into train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE = 'framingham_v2.csv'

data = pd.read_csv(PATH + FILE)

# Inspect data.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(data.head())
print(data.describe())

# Split the data into train and test sets.
X_train, X_test, y_train, y_test = get_train_and_test_data(data)

# Show the distribution of the target variable in training and test data.
print("\n ***Distribution of target variable in training and test data.")
print("Training data:")
print(y_train['TenYearCHD'].value_counts())
print("Test data:")
print(y_test['TenYearCHD'].value_counts())

# Perform logistic regression.
logistic_regression = LogisticRegression(solver='newton-cg', max_iter=1000)
logistic_regression.fit(X_train, y_train)
evaluate_model(X_test, y_test, y_train, logistic_regression, "Before SCUT")

# showYPlots(y_train, y_test, "Before SCUT")

# Perform SCUT.
dfTrain = X_train.copy()
dfTrain['TenYearCHD'] = y_train
scut = SCUT()
df_scut = scut.balance(dfTrain, 'TenYearCHD')

# Adjust y_train and X_train with better represented minority.
y_train = df_scut['TenYearCHD']
X_train = df_scut
del df_scut['TenYearCHD']

# Perform logistic regression with SCUT-treated train data and
# evaluate with untouched test data.
clf = LogisticRegression(solver='newton-cg', max_iter=1000)
clf.fit(X_train, y_train)
evaluate_model(X_test, y_test, y_train, clf, "After SCUT")

# showYPlots(y_train, y_test, "After SCUT")

# Perform logistic regression with SCUT-treated train data and MinMaxScaler
minmax_scaler = MinMaxScaler()

X_train_scaled_minmax = minmax_scaler.fit_transform(X_train)
X_test_scaled_minmax = minmax_scaler.transform(X_test)

clf = LogisticRegression(solver='newton-cg', max_iter=1000)
clf.fit(X_train_scaled_minmax, y_train)
evaluate_model(X_test_scaled_minmax, y_test, y_train, clf, "After SCUT and MinMaxScaler")

# Perform logistic regression with SCUT-treated train data and StandardScaler
standard_scaler = StandardScaler()

X_train_scaled_standard = standard_scaler.fit_transform(X_train)
X_test_scaled_standard = standard_scaler.transform(X_test)

clf = LogisticRegression(solver='newton-cg', max_iter=1000)
clf.fit(X_train_scaled_standard, y_train)
evaluate_model(X_test_scaled_standard, y_test, y_train, clf, "After SCUT and StandardScaler")

# Perform logistic regression with SCUT-treated train data and RobustScaler
robust_scaler = RobustScaler()

X_train_scaled_robust = robust_scaler.fit_transform(X_train)
X_test_scaled_robust = robust_scaler.transform(X_test)

clf = LogisticRegression(solver='newton-cg', max_iter=1000)
clf.fit(X_train_scaled_robust, y_train)
evaluate_model(X_test_scaled_robust, y_test, y_train, clf, "After SCUT and RobustScaler")