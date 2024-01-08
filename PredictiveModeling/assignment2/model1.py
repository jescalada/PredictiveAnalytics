import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score, roc_curve, \
    recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from crucio import SCUT
import time
import numpy as np

from sklearn.preprocessing import RobustScaler

start_time = time.perf_counter()

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "\\..\\datasets\\"
PATH = ROOT_PATH + DATASET_DIR
FILE = 'CustomerChurn.csv'

dataframe = pd.read_csv(PATH + FILE)

dataframe = dataframe.sample(20000)

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

# Drop ID column.
dataframe.drop('CustomerID', axis=1, inplace=True)


# Keep only the following columns: AccountAge, ViewingHoursPerWeek, GenrePreference and UserRating
dataframe = dataframe[['AccountAge', 'ViewingHoursPerWeek', 'GenrePreference', 'UserRating', 'Churn']]

### Impute missing values using random values from a normal distribution ###

rng = np.random.default_rng()
# Define AccountAge mean and standard deviation for users who churned
account_age_mean_churned = dataframe[dataframe['Churn'] == 1]['AccountAge'].mean()
account_age_std_churned = dataframe[dataframe['Churn'] == 1]['AccountAge'].std()

# Define AccountAge mean and standard deviation for users who did not churn
account_age_mean_not_churned = dataframe[dataframe['Churn'] == 0]['AccountAge'].mean()
account_age_std_not_churned = dataframe[dataframe['Churn'] == 0]['AccountAge'].std()

# Define ViewingHoursPerWeek mean and standard deviation
viewing_hours_per_week_mean = dataframe['ViewingHoursPerWeek'].mean()
viewing_hours_per_week_std = dataframe['ViewingHoursPerWeek'].std()

# Iterate through each row and impute missing values
for index, row in dataframe.iterrows():
    if pd.isnull(row['AccountAge']):
        if row['Churn'] == 1:
            dataframe.at[index, 'AccountAge'] = rng.normal(account_age_mean_churned, account_age_std_churned)
        else:
            dataframe.at[index, 'AccountAge'] = rng.normal(account_age_mean_not_churned, account_age_std_not_churned)
    if pd.isnull(row['ViewingHoursPerWeek']):
        dataframe.at[index, 'ViewingHoursPerWeek'] = rng.normal(viewing_hours_per_week_mean, viewing_hours_per_week_std)

# Bin the AccountAge column into 10 bins of width 10
dataframe['AccountAgeBins'] = pd.cut(dataframe['AccountAge'], bins=10, labels=False)

# Round user ratings to nearest 0.1
dataframe['UserRating'] = dataframe['UserRating'].round(1)

# Bin the UserRating column into 4 bins of width 1
dataframe['UserRatingBins'] = pd.cut(dataframe['UserRating'], bins=4)


# Use RobustScaler to scale the numerical columns
scaler = RobustScaler()
dataframe[['AccountAge', 'ViewingHoursPerWeek', 'UserRating']] = scaler.fit_transform(
    dataframe[['AccountAge', 'ViewingHoursPerWeek', 'UserRating']]
)
print("After scaling:")
print(dataframe.head())

# Add dummies for categorical columns
dataframe = pd.get_dummies(dataframe, columns=['GenrePreference', 'AccountAgeBins', 'UserRatingBins'])
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

print("Pre-SCUT:")
print(f"X_train head: \n{X_train.head()}")
print(f"y_train head: \n{y_train.head()}")

# Perform SCUT.
dfTrain = X_train.copy()
dfTrain['Churn'] = y_train

scut = SCUT(
    k=5,
    binary_columns=['GenrePreference_Action', 'GenrePreference_Comedy', 'GenrePreference_Drama', 'GenrePreference_Fantasy',
                            'GenrePreference_Sci-Fi', 'AccountAgeBins_0', 'AccountAgeBins_1',
                                               'AccountAgeBins_2', 'AccountAgeBins_3', 'AccountAgeBins_4',
                                               'AccountAgeBins_5', 'AccountAgeBins_6', 'AccountAgeBins_7',
                                               'AccountAgeBins_8', 'AccountAgeBins_9', 'UserRatingBins_(0.996, 2.0]',
                                               'UserRatingBins_(2.0, 3.0]', 'UserRatingBins_(3.0, 4.0]', 'UserRatingBins_(4.0, 5.0]', 'Churn']
)
df_scut = scut.balance(dfTrain, 'Churn')

# Adjust y_train and X_train with better represented minority.
y_train = df_scut['Churn']
X_train = df_scut
del df_scut['Churn']

# Change target column to binary
y_train = y_train.astype('int')

print("Post-SCUT:")
print(f"X_train head: \n{X_train.head()}")
print(f"y_train head: \n{y_train.head()}")

# Create a Logistic Regression model
model = LogisticRegression(max_iter=10000, solver='lbfgs', class_weight='balanced')

# Fit the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)


# Perform cross-validation
# scores = cross_val_score(model, X, y, cv=5)
# print(scores)

auc = roc_auc_score(y_test, y_prob[:, 1],)
print('Logistic: ROC AUC=%.3f' % (auc))

# calculate roc curves
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_prob[:, 1])
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.plot([0, 1], [0, 1], '--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

end_time = time.perf_counter()
print(f"Runtime of the program is {end_time - start_time:.3f} seconds")

# Cross-validate the model
accuracy = cross_val_score(model, X, y, cv=5)
f1 = cross_val_score(model, X, y, cv=5, scoring='f1')
precision = cross_val_score(model, X, y, cv=5, scoring='precision')
recall = cross_val_score(model, X, y, cv=5, scoring='recall')

print(f"Accuracy: {accuracy.mean():.3f} +/- {accuracy.std():.3f}")
print(f"F1: {f1.mean():.3f} +/- {f1.std():.3f}")
print(f"Precision: {precision.mean():.3f} +/- {precision.std():.3f}")
print(f"Recall: {recall.mean():.3f} +/- {recall.std():.3f}")
