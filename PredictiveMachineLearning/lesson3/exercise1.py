import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE = 'housing_classification.csv'

# Get the housing data
df = pd.read_csv(PATH + FILE)
# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head(5))

# Split into two sets
y = df['price']
X = df.drop('price', axis=1)


# Create classifiers
knn = KNeighborsClassifier()
svc = SVC()
rg = RidgeClassifier()
lr = LogisticRegression(fit_intercept=True, solver='liblinear')

# Build array of classifiers.
classifier_array = [knn, svc, rg, lr]


def show_stats(classifier, scores):
    print(classifier + ":    ", end="")
    str_mean = str(round(scores.mean(), 2))

    str_std = str(round(scores.std(), 2))
    print("Mean: " + str_mean + "   ", end="")
    print("Std: " + str_std)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


def evaluate_model(model, X_test, y_test, title):
    print("\n*** " + title + " ***")
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    print(report)


# Search for the best classifier.
for clf in classifier_array:
    modelType = clf.__class__.__name__

    # Create and evaluate stand-alone model.
    clfModel = clf.fit(X_train, y_train)
    evaluate_model(clfModel, X_test, y_test, modelType)

    # max_features means the maximum number of features to draw from X.
    # max_samples sets the percentage of available data used for fitting.
    bagging_clf = BaggingClassifier(clf, max_samples=0.4, max_features=6,
                                    n_estimators=100)
    bagged_model = bagging_clf.fit(X_train, y_train)
    evaluate_model(bagged_model, X_test, y_test, "Bagged: " + modelType)
