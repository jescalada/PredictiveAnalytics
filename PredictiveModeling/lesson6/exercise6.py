import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "bank-additional-full.csv"
df = pd.read_csv(PATH + CSV_DATA,
                 skiprows=1,  # Don't include header row as part of data.
                 encoding="ISO-8859-1", sep=';',
                 names=(
                     "age", "job", "marital", "education", "default", "housing", "loan", "contact",
                     "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome",
                     "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"))
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(df.head())
print(df.describe().transpose())
print(df.info())

targetList = []
for i in range(0, len(df)):
    if df.loc[i]['y'] == 'yes':
        targetList.append(1)
    else:
        targetList.append(0)
df['target'] = targetList

temp_df = df[["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week",
              "poutcome"]]  # Isolate columns
dummy_df = pd.get_dummies(temp_df, columns=["job", "marital", "education", "default",
                                            "housing", "loan", "contact", "month", "day_of_week",
                                            "poutcome"])  # Get dummies
df = pd.concat(([df, dummy_df]), axis=1)  # Join dummy df with original df

X = df[[
    "campaign", "cons.price.idx", "cons.conf.idx",
    "job_admin.",
    "job_entrepreneur",
    "job_services", "job_unemployed",
    "marital_married",
    "education_basic.6y", "education_basic.9y",
    "education_unknown", "month_jul",
    "day_of_week_mon", "day_of_week_thu",
    "poutcome_failure" ]]
y = df[['target']]
print(X.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

# Use RFE to select the best features
logistic_regression = LogisticRegression(solver='liblinear', max_iter=1000)
rfe = RFE(logistic_regression, n_features_to_select=15)
rfe = rfe.fit(X_train, y_train.values.ravel())

# Show top features.
for i in range(0, len(X.keys())):
    if rfe.support_[i]:
        print(X.keys()[i])

# Show chi-square scores for each feature.
# There is 1 degree freedom since 1 predictor during feature evaluation.
# Generally, >=3.8 is good)

test = SelectKBest(score_func=chi2, k=15)

# Use scaled data to fit KBest
XScaled = MinMaxScaler().fit_transform(X)
chiScores = test.fit(XScaled, y)  # Summarize scores
np.set_printoptions(precision=3)

# Search here for insignificant features.
print("\nPredictor Chi-Square Scores: " + str(chiScores.scores_))

# Create a sorted list of the top features.
dfFeatures = pd.DataFrame()
for i in range(0, len(chiScores.scores_)):
    headers = list(X.keys())
    featureObject = {"feature": headers[i], "chi-square score": chiScores.scores_[i]}
    dfFeatures = dfFeatures._append(featureObject, ignore_index=True)

print("\nTop Features")
dfFeatures = dfFeatures.sort_values(by=['chi-square score'], ascending=False)
print(dfFeatures.tail(40))

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
def buildAndEvaluateClassifier(features, X, y):
    # Re-assign X with significant columns only after chi-square test.
    X = X[features]

    # Split data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Perform logistic regression.
    logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear')

    # Fit the model.
    logisticModel.fit(X_train, y_train)
    y_pred = logisticModel.predict(X_test)
    y_prob = logisticModel.predict_proba(X_test)
    # print(y_pred)

    # Show accuracy scores.
    print('Results without scaling:')

    print("Recall:    " + str(recall_score(y_test, y_pred)))
    print("Precision: " + str(precision_score(y_test, y_pred)))
    print("F1:        " + str(f1_score(y_test, y_pred)))
    print("Accuracy:  " + str(accuracy_score(y_test, y_pred)))
    return X_test, y_test, y_pred, y_prob

X_test, y_test, y_pred, y_prob =\
buildAndEvaluateClassifier(dfFeatures['feature'], X, y)


from sklearn.metrics           import roc_curve
from sklearn.metrics           import roc_auc_score

auc = roc_auc_score(y_test, y_prob[:, 1],)
print('Logistic: ROC AUC=%.3f' % (auc))

# calculate roc curves
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_prob[:, 1])
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.plot([0,1], [0,1], '--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

print(df['target'].value_counts())
