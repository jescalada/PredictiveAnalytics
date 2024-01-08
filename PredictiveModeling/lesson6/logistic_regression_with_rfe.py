import warnings
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os

from sklearn.model_selection import train_test_split

warnings.simplefilter(action="ignore", category=FutureWarning)

# Read the data:
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
df = pd.read_csv(PATH + "Divorce.csv", header=0)

# Separate the target and independent variable
X = df.copy()  # Create separate copy to prevent unwanted tampering of data.
del X['Divorce']  # Delete target variable.

# Target variable
y = df['Divorce']

# Create the object of the model
model = LogisticRegression()

# Specify the number of  features to select
rfe = RFE(model, n_features_to_select=8)

# fit the model
rfe = rfe.fit(X, y)


def build_and_evaluate_classifier(features, X, y):
    # Re-assign X with significant columns only after chi-square test.
    X = X[features]

    # Split data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Perform logistic regression.
    logistic_model = LogisticRegression(fit_intercept=True, solver='liblinear', random_state=0)

    # Fit the model.
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)
    # print(y_pred)

    # Show accuracy scores.
    print('Results without scaling:')

    # Show confusion matrix
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print("\nConfusion Matrix")
    print(cm)

    print("Recall:    " + str(recall_score(y_test, y_pred)))
    print("Precision: " + str(precision_score(y_test, y_pred)))
    print("F1:        " + str(f1_score(y_test, y_pred)))
    print("Accuracy:  " + str(accuracy_score(y_test, y_pred)))


features = ['Q3', 'Q6', 'Q17', 'Q18', 'Q26', 'Q39', 'Q40', 'Q49']
build_and_evaluate_classifier(features, X, y)
