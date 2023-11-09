from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "\\..\\datasets\\"
CSV_DATA = "milk.csv"
df = pd.read_csv(ROOT_PATH + DATASET_DIR + CSV_DATA)

# Split the data at the start to hold back a test set.
train, test = train_test_split(df, test_size=0.2)

X_train = train.copy()
X_test = test.copy()
del X_train['labels']
del X_test['labels']
del X_train['dates']
del X_test['dates']

y_train = train['labels']
y_test = test['labels']

# Scale X values.
xscaler = StandardScaler()
Xtrain_scaled = xscaler.fit_transform(X_train)
Xtest_scaled = xscaler.transform(X_test)

# Generate PCA components.
pca = PCA(0.95)

# Always fit PCA with train data. Then transform the train data.
X_reduced_train = pca.fit_transform(Xtrain_scaled)

# Transform test data with PCA
X_reduced_test = pca.transform(Xtest_scaled)

print("\nPrincipal Components")
print(pca.components_)
print("\nPrincipal Components shape")
print(pca.components_.shape)

print("\nExplained variance: ")
print(pca.explained_variance_)

# Train regression model on training data
model = LogisticRegression(solver='liblinear')
model.fit(X_reduced_train, y_train)

# Predict with test data.
preds = model.predict(X_reduced_test)

report = classification_report(y_test, preds)
print(report)

import matplotlib.pyplot as plt

eig_vals = pca.explained_variance_

# Show the scree plot.
plt.plot([1, 2, 3], eig_vals, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.show()

# Calculate cumulative values.
sumEigenvalues = eig_vals.sum()
cumulativeValues = []
cumulativeSum = 0
for i in range(0, len(eig_vals) + 1):
    cumulativeValues.append(cumulativeSum)
    if i < len(eig_vals):
        cumulativeSum += eig_vals[i] / sumEigenvalues

# Show cumulative variance plot.
import matplotlib.pyplot as plt

plt.plot([0, 1, 2, 3], cumulativeValues, 'ro-', linewidth=2)
plt.title('Variance Explained by Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.show()
