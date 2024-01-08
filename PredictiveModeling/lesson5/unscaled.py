import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "computerPurchase.csv"
df = pd.read_csv(PATH + CSV_DATA)

# Separate into x and y values.
X = df[["Age", "EstimatedSalary"]]
y = df['Purchased']

# Split data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)

# Perform logistic regression.
logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear')

# Fit the model.
logisticModel.fit(X_train, y_train)
y_pred = logisticModel.predict(X_test)

# Show confusion matrix and accuracy scores.
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

print('\nAccuracy: ', metrics.accuracy_score(y_test, y_pred))
print("\nConfusion Matrix")
print(cm)
