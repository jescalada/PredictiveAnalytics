import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, recall_score, precision_score, \
    accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MaxAbsScaler

start_time = time.perf_counter()

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "\\..\\datasets\\"
PATH = ROOT_PATH + DATASET_DIR
FILE = 'CustomerChurn.csv'

dataframe = pd.read_csv(PATH + FILE)

# Select all the rows with churned users
churned = dataframe[dataframe['Churn'] == 1]
not_churned = dataframe[dataframe['Churn'] == 0]

# Sample the not_churned dataframe to match the number of churned users
not_churned = not_churned.sample(len(churned))

# Combine the churned and not_churned dataframes
dataframe = churned._append(not_churned)

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

# Drop ID column.
dataframe.drop('CustomerID', axis=1, inplace=True)


from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
dataframe['AccountAge'] = imputer.fit_transform(dataframe[['AccountAge']])
dataframe['ViewingHoursPerWeek'] = imputer.fit_transform(dataframe[['ViewingHoursPerWeek']])
dataframe['AverageViewingDuration'] = imputer.fit_transform(dataframe[['AverageViewingDuration']])

# Bin the AccountAge column into 10 bins of width 10
dataframe['AccountAgeBins'] = pd.cut(dataframe['AccountAge'], bins=10, labels=False)

# Round user ratings to nearest 0.1
dataframe['UserRating'] = dataframe['UserRating'].round(1)

# Bin the UserRating column into 4 bins of width 1
dataframe['UserRatingBins'] = pd.cut(dataframe['UserRating'], bins=4)

# Use RobustScaler to scale the numerical columns
scaler = MaxAbsScaler()
dataframe[['AccountAge', 'MonthlyCharges', 'ViewingHoursPerWeek', 'AverageViewingDuration',
           'ContentDownloadsPerMonth', 'SupportTicketsPerMonth']] = scaler.fit_transform(
    dataframe[['AccountAge', 'MonthlyCharges', 'ViewingHoursPerWeek', 'AverageViewingDuration',
               'ContentDownloadsPerMonth', 'SupportTicketsPerMonth']])

# Add dummies for categorical columns
dataframe = pd.get_dummies(dataframe, columns=['SubscriptionType', 'PaymentMethod', 'PaperlessBilling', 'ContentType',
                                               'MultiDeviceAccess', 'DeviceRegistered', 'GenrePreference', 'Gender',
                                               'ParentalControl', 'SubtitlesEnabled',
                                               'AccountAgeBins', 'UserRatingBins'])

# Keep only the columns found through RFE:
dataframe = dataframe[['AccountAge',
                       'MonthlyCharges',
                       'ViewingHoursPerWeek',
                       'AverageViewingDuration',
                       'ContentDownloadsPerMonth',
                       'SupportTicketsPerMonth',
                       'PaymentMethod_Mailed check',
                       'PaperlessBilling_No',
                       'ContentType_TV Shows',
                       'MultiDeviceAccess_No',
                       'Churn']]

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
model = LogisticRegression(max_iter=10000, n_jobs=4, solver='newton-cholesky')

# Fit the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

auc = roc_auc_score(y_test, y_prob[:, 1], )
print('Logistic: ROC AUC=%.3f' % (auc))

# Calculate roc curves
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_prob[:, 1])
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.plot([0, 1], [0, 1], '--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Calculate the mean F1 score
print("Mean F1 score: ")
f1 = cross_val_score(model, X, y, cv=5, scoring='f1', n_jobs=4)
print(f"F1 mean: {f1.mean()}")
print(f"F1 std: {f1.std()}")

# Calculate the mean AUC score
print("Mean AUC score: ")
print(cross_val_score(model, X, y, cv=5, scoring='roc_auc', n_jobs=4).mean())

# Calculate the mean precision score
print("Mean precision score: ")
print(cross_val_score(model, X, y, cv=5, scoring='precision').mean())

# Calculate the mean recall score
print("Mean recall score: ")
print(cross_val_score(model, X, y, cv=5, scoring='recall').mean())

# Print the confusion matrix
print("Confusion matrix: ")
print(confusion_matrix(y_test, y_pred))

dataframe_y = dataframe['Churn']
dataframe_x = dataframe.drop('Churn', axis=1)

# Perform KFold validation
print("KFold validation: ")
kf = KFold(n_splits=5)
kf.get_n_splits(dataframe)
KFold(n_splits=5, random_state=None, shuffle=True)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
for train_index, test_index in kf.split(dataframe_x):
    X_train = dataframe_x.loc[dataframe_x.index.intersection(train_index), :]
    X_test = dataframe_x.loc[dataframe_x.index.intersection(test_index), :]
    y_train = dataframe_y.loc[dataframe_y.index.intersection(train_index)]
    y_test = dataframe_y.loc[dataframe_y.index.intersection(test_index)]

    # Build model
    model = LogisticRegression(max_iter=10000, solver='newton-cholesky')
    model.fit(X_train, y_train)

    # Evaluate model
    f1 = f1_score(y_test, model.predict(X_test))
    accuracy = accuracy_score(y_test, model.predict(X_test))
    precision = precision_score(y_test, model.predict(X_test))
    recall = recall_score(y_test, model.predict(X_test))

    f1_scores.append(f1_score(y_test, model.predict(X_test)))
    accuracy_scores.append(accuracy_score(y_test, model.predict(X_test)))
    precision_scores.append(precision_score(y_test, model.predict(X_test)))
    recall_scores.append(recall_score(y_test, model.predict(X_test)))

    print(f"Confusion matrix:\n{confusion_matrix(y_test, model.predict(X_test))}")

print(f"Mean Accuracy: {np.mean(accuracy_scores)}")
print(f"SD Accuracy: {np.std(accuracy_scores)}")

print(f"Mean Precision: {np.mean(precision_scores)}")
print(f"SD Precision: {np.std(precision_scores)}")

print(f"Mean Recall: {np.mean(recall_scores)}")
print(f"SD Recall: {np.std(recall_scores)}")

print(f"Mean F1: {np.mean(f1_scores)}")
print(f"SD F1: {np.std(f1_scores)}")

# Save model and scaler into a pickle file
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Finish the timer
end_time = time.perf_counter()
print(f"Runtime of the program is {end_time - start_time:.3f} seconds")

# Show coefficients with their corresponding features
print("Coefficients: ")
print(pd.DataFrame(zip(X.columns, np.transpose(model.coef_))))