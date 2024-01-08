import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler

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
    "age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed", "job_admin.", "job_blue-collar",
    "job_entrepreneur", "job_housemaid", "job_management", "job_retired",
    "job_self-employed", "job_services", "job_student", "job_technician", "job_unemployed",
    "job_unknown", "marital_divorced", "marital_married", "marital_single",
    "marital_unknown", "education_basic.4y", "education_basic.6y", "education_basic.9y",
    "education_high.school", "education_illiterate", "education_professional.course",
    "education_university.degree", "education_unknown", "default_no",
    "default_unknown", "default_yes", "housing_no", "housing_unknown", "housing_yes",
    "loan_no", "loan_unknown", "loan_yes", "contact_cellular", "contact_telephone",
    "month_apr", "month_aug", "month_dec", "month_jul", "month_jun", "month_mar",
    "month_may", "month_nov", "month_oct", "month_sep", "day_of_week_fri",
    "day_of_week_mon", "day_of_week_thu", "day_of_week_tue", "day_of_week_wed",
    "poutcome_failure", "poutcome_nonexistent", "poutcome_success", ]]
y = df[['target']]
print(X.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

# Use RFE to select the best features
logistic_regression = LogisticRegression(solver='liblinear', max_iter=1000)
rfe = RFE(logistic_regression, n_features_to_select=15)
rfe = rfe.fit(X_train, y_train.values.ravel())

print("\nTop Features based on Recursive Feature Elimination:")
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

print("\nTop Features based on Chi-Square Scores:")
dfFeatures = dfFeatures.sort_values(by=['chi-square score'], ascending=False)
print(dfFeatures.tail(40))
