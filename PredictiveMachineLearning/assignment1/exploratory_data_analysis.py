import pandas as pd

# Load training data from train.csv
pd.set_option('display.max_columns', None)
df = pd.read_csv('train.csv')

# Info
print(df.head())
print(df.info())
print(df.describe())

# Plot correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=2)
plt.title('Correlation Matrix between variables')
plt.show()

# Create 4 boxplots comparing the distribution of 'ram' and 'battery_power' for each 'price_range'
plt.figure(figsize=(10, 10))
sns.boxplot(x='price_range', y='ram', data=df)
plt.title('Boxplot of RAM by Price Range')
plt.show()

plt.figure(figsize=(10, 10))
sns.boxplot(x='price_range', y='battery_power', data=df)
plt.title('Boxplot of Battery Power by Price Range')
plt.show()

plt.figure(figsize=(10, 10))
sns.boxplot(x='price_range', y='px_width', data=df)
plt.title('Boxplot of Width by Price Range')
plt.show()

plt.figure(figsize=(10, 10))
sns.boxplot(x='price_range', y='px_height', data=df)
plt.title('Boxplot of Height by Price Range')
plt.show()

# Create a scatterplot of 'ram' and 'battery_power' with the color of the points determined by the 'price_range'
plt.figure(figsize=(10, 10))
sns.scatterplot(x='ram', y='battery_power', hue='price_range', data=df)
plt.title('Scatterplot of RAM and Battery Power by Price Range')
plt.show()

# Create a scatterplot of 'fc' and 'pc' with the color of the points determined by the 'price_range'
plt.figure(figsize=(10, 10))
sns.scatterplot(x='fc', y='pc', hue='price_range', data=df)
plt.title('Scatterplot of Front Camera and Primary Camera by Price Range')
plt.show()

# Perform recursive feature elimination to select the best features
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# Create a logistic regression model
estimator = LogisticRegression()

X = df.copy()
del X['price_range']

y = df['price_range']

features = RFE(estimator, n_features_to_select=5)
features = features.fit(X, y)
print(features.support_)
print(features.ranking_)

# Print the names of the best features using the list of boolean values
for i in range(len(features.support_)):
    if features.support_[i]:
        print(X.columns[i])

# Perform Forward Feature Selection to select the best features
from mlxtend.feature_selection import SequentialFeatureSelector

# Create a forward feature selector
selector = SequentialFeatureSelector(estimator, k_features=5, forward=True, scoring='accuracy', cv=4)
selector = selector.fit(X, y)

# Print the names of the best features
print(selector.k_feature_names_)
print(selector.k_score_)
print(selector.subsets_)

# Perform Feature Importance using Random Forest
from sklearn.ensemble import RandomForestClassifier

# Create a random forest model
model = RandomForestClassifier(n_estimators=100)

# Fit the model
model.fit(X, y)

# Print the feature importances
print(model.feature_importances_)
print(X.columns)

# Print the top 5 features
import numpy as np

indices = np.argsort(model.feature_importances_)[::-1]
for i in range(5):
    print(X.columns[indices[i]])
