# Import scikit-learn dataset library
import numpy as np
from sklearn import datasets

# Load dataset
from sklearn.metrics import mean_squared_error

iris = datasets.load_iris()

# Creating a DataFrame of given iris dataset.
import pandas as pd

data = pd.DataFrame({
    'sepal length': iris.data[:, 0],
    'sepal width': iris.data[:, 1],
    'petal length': iris.data[:, 2],
    'petal width': iris.data[:, 3],
    'species': iris.target
})
print(data.head())

# Import train_test_split function
from sklearn.model_selection import train_test_split

feature_list = ['sepal length', 'sepal width', 'petal length', 'petal width']

X = data[feature_list]  # Features
y = data['species']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Create a Gaussian Classifier
rf = RandomForestClassifier(n_estimators=100)

# Train the model using the training sets y_pred=rf.predict(X_test)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Predict species for a single flower.
# sepal length = 3, sepal width = 5
# petal length = 4, petal width = 2
prediction = rf.predict([[3, 5, 4, 2]])
# 'setosa', 'versicolor', 'virginica'
print(prediction)

# Get numerical feature importances
importances = list(rf.feature_importances_)


def show_feature_importances(importances, feature_list):
    df_importance = pd.DataFrame()
    for i in range(0, len(importances)):
        df_importance = df_importance._append({"importance": importances[i],
                                               "feature": feature_list[i]},
                                              ignore_index=True)

    df_importance = df_importance.sort_values(by=['importance'],
                                              ascending=False)
    print(df_importance)


show_feature_importances(importances, feature_list)

X = data[feature_list]  # Features

X = X.drop('sepal width', axis=1)
y = data['species']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Create a Gaussian Classifier
rf = RandomForestClassifier(n_estimators=100)

# Train the model using the training sets y_pred=rf.predict(X_test)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
