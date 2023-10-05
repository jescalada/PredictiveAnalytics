import numpy as np
import pandas as pd
import os

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE = 'employee_turnover.csv'
df = pd.read_csv(FOLDER_PATH + FILE)
print(df)

# Separate into x and y values.
predictor_variables = list(df.keys())
predictor_variables.remove('turnover')
print(predictor_variables)

# Create X and y values.
X = df[predictor_variables]
y = df['turnover']

# Perform Chi-Square test to determine the most statistically significant variables
test = SelectKBest(score_func=chi2, k=3)
chi_scores = test.fit(X, y)  # Summarize scores
np.set_printoptions(precision=3)

print("\nPredictor variables: " + str(predictor_variables))
print("Predictor Chi-Square Scores: " + str(chi_scores.scores_))

# Another technique for showing the most statistically
# significant variables involves the get_support() function.
cols = chi_scores.get_support(indices=True)
print(cols)
features = X.columns[cols]
print(np.array(features))

# Re-assign X with significant columns only after chi-square test.
X = df[['experience', 'industry', 'way']]

# Split data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=0)

# Build logistic regression model and make predictions.
logistic_model = LogisticRegression(fit_intercept=True, solver='liblinear',
                                    random_state=0)
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
print(y_pred)

# Evaluate model using confusion matrix
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / \
           (confusion_matrix[0][0] + confusion_matrix[0][1] +
            confusion_matrix[1][0] + confusion_matrix[1][1])
print(f'Accuracy: {accuracy}')
