import pandas as pd
import os
from sklearn.model_selection import train_test_split

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV = "bill_authentication.csv"
dataset = pd.read_csv(PATH + CSV)
X = dataset.drop('Class', axis=1)
y = dataset['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20, random_state=0)


def manually_classify(X, y):
    predictions = []
    ones = 0
    zeros = 0
    for i in range(0, len(X)):

        if (X.iloc[i]['Variance'] <= 0.274):  # Blue
            if (X.iloc[i]['Skewness'] <= 7.565):
                ones += 1  # Blue
            else:
                zeros += 1
        else:
            if (X.iloc[i]['Kurtosis'] <= -4.394):
                ones += 1  # Blue
            else:
                zeros += 1

    print("Zeros: " + str(zeros))
    print("Ones: " + str(ones))


manually_classify(X_train, y_train)
print(len(X_train))
