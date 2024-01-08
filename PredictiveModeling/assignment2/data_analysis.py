import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tools
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "\\..\\datasets\\"
PATH = ROOT_PATH + DATASET_DIR
FILE = 'CustomerChurn.csv'

dataframe = pd.read_csv(PATH + FILE)

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

print(dataframe.head())
print(dataframe.describe())

# Replace missing values in AccountAge with the mean
dataframe['AccountAge'].fillna(dataframe['AccountAge'].mean(), inplace=True)

# Replace missing values in ViewingHoursPerWeek with the mean
dataframe['ViewingHoursPerWeek'].fillna(dataframe['ViewingHoursPerWeek'].mean(), inplace=True)

# Replace missing values in AverageViewingDuration with the mean
dataframe['AverageViewingDuration'].fillna(dataframe['AverageViewingDuration'].mean(), inplace=True)

# # Bin the AccountAge column into 10 bins of width 10
dataframe['AccountAgeBins'] = pd.cut(dataframe['AccountAge'], bins=10, labels=False)
# print(dataframe.head())
#
# # Bin the UserRating column into 4 bins of width 1
dataframe['UserRatingBins'] = pd.cut(dataframe['UserRating'], bins=4)

# Make a simple Logistic Regression model
# Impute missing numeric values using KNNImputer
imputer = KNNImputer(n_neighbors=5)


# Add dummies for categorical columns
dataframe = dataframe.drop(columns=['CustomerID'])
dataframe = pd.get_dummies(dataframe, columns=['SubscriptionType', 'PaymentMethod', 'PaperlessBilling', 'ContentType',
                                               'MultiDeviceAccess', 'DeviceRegistered', 'GenrePreference', 'Gender',
                                               'ParentalControl', 'SubtitlesEnabled', 'AccountAgeBins', 'UserRatingBins'])

# replace NaN values with the mean of the 5 nearest neighbors
# dataframe = pd.DataFrame(imputer.fit_transform(dataframe), columns=dataframe.columns)

print(dataframe.head())

X = dataframe.copy()

statsmodels.tools.add_constant(X)

# Remove the target column from the dataframe
X.drop(columns=['Churn'], inplace=True)

y = dataframe['Churn']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)

# Create a Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Fit the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate the accuracy score on the test set
print("Accuracy score: ")
print(accuracy_score(y_test, predictions))

# Calculate the precision
print("Precision: ")
print(precision_score(y_test, predictions))

# Calculate ROC
roc = roc_auc_score(y_test, predictions)
print("ROC: ")
print(roc)

# Print the confusion matrix
print("Confusion matrix: ")
print(confusion_matrix(y_test, predictions))

# Print the F1 score
print("F1 score: ")
print(f1_score(y_test, predictions))

# Perform cross-validation
# scores = cross_val_score(model, X, y, cv=5)
# print(scores)
