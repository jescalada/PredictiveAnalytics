import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import os
from sklearn import svm

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"

# load the dataset
df = pd.read_csv(PATH + 'diabetes.csv', sep=',')
# split into input (X) and output (y) variables

X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age']]
y = df[['Outcome']]
# Split into train and test data sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Perform logistic regression.
logistic_model = LogisticRegression(fit_intercept=True, random_state=0,
                                    solver='liblinear')
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

# Show model coefficients and intercept.
print("\nModel Coefficients: ")
print("\nIntercept: ")
print(logistic_model.intercept_)

print(logistic_model.coef_)

# Show confusion matrix and accuracy scores.
confusion_matrix = pd.crosstab(np.array(y_test['Outcome']), y_pred,
                               rownames=['Actual'],
                               colnames=['Predicted'])

print('\nAccuracy: ', metrics.accuracy_score(y_test, y_pred))
print("\nConfusion Matrix")
print(confusion_matrix)

# Create a svm Classifier using one of the following options:
# linear, polynomial, and radial
clf = svm.SVC(kernel='linear')

# Train the model using the training set.
clf.fit(X_train, y_train)

# Evaluate the model.
y_pred = clf.predict(X_test)
from sklearn import metrics

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
