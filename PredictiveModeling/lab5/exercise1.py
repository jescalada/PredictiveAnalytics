import numpy as np
import pandas as pd
import os

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

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
