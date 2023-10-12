import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import RobustScaler
import os

import numpy as np
from sklearn.model_selection import KFold

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "computerPurchase.csv"
df = pd.read_csv(PATH + CSV_DATA, sep=',')

# prepare cross validation with three folds and 1 as a random seed.
kfold = KFold(n_splits=3, shuffle=True)

accuracy_list = []
precision_list = []
recall_list = []

fold_count = 1


def get_test_and_train_data(train_indexes, test_indexes, df):
    df_train = df.iloc[train_indexes, :]  # Gets all rows with train indexes.
    df_test = df.iloc[train_indexes, :]

    x_train = df_train[['EstimatedSalary', 'Age']]
    x_test = df_test[['EstimatedSalary', 'Age']]
    y_train = df_train[['Purchased']]
    y_test = df_test[['Purchased']]
    return x_train, x_test, y_train, y_test


for trainIdx, testIdx in kfold.split(df):
    X_train, X_test, y_train, y_test = \
        get_test_and_train_data(trainIdx, testIdx, df)

    # Recommended to only fit on training data.
    # Scaling only needed for X since y ranges between 0 and 1.
    scaler_X = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)  # Fit and transform.
    X_test_scaled = scaler_X.transform(X_test)  # Transform only.

    # Perform logistic regression.
    logistic_model = LogisticRegression(fit_intercept=True,
                                        solver='liblinear')
    # Fit the model.
    logistic_model.fit(X_train_scaled, y_train)

    y_pred = logistic_model.predict(X_test_scaled)
    y_prob = logistic_model.predict_proba(X_test_scaled)

    # Show confusion matrix and accuracy scores.
    y_test_array = np.array(y_test['Purchased'])
    cm = pd.crosstab(y_test_array, y_pred, rownames=['Actual'],
                     colnames=['Predicted'])

    print("\n***K-fold: " + str(fold_count))
    fold_count += 1

    accuracy = metrics.accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    print('\nAccuracy: ', accuracy)

    precision = metrics.precision_score(y_test, y_pred)
    precision_list.append(precision)
    print('\nPrecision: ', precision)

    recall = metrics.recall_score(y_test, y_pred)
    recall_list.append(recall)
    print('\nRecall: ', recall)

    print("\nConfusion Matrix")
    print(cm)

    from sklearn.metrics import classification_report, roc_auc_score

    print(classification_report(y_test, y_pred))

    from sklearn.metrics import average_precision_score

    average_precision = average_precision_score(y_test, y_pred)

    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))

    # calculate scores
    auc = roc_auc_score(y_test, y_prob[:, 1], )
    print('Logistic: ROC AUC=%.3f' % (auc))

print("\nAccuracy and Standard Deviation For All Folds:")
print("*********************************************")
print("Average accuracy: " + str(np.mean(accuracy_list)))
print("Accuracy std: " + str(np.std(accuracy_list)))

print("\nPrecision and Standard Deviation For All Folds:")
print("*********************************************")
print("Average precision: " + str(np.mean(precision_list)))
print("Precision std: " + str(np.std(precision_list)))

print("\nRecall and Standard Deviation For All Folds:")
print("*********************************************")
print("Average recall: " + str(np.mean(recall_list)))
print("Recall std: " + str(np.std(recall_list)))