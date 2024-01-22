import pandas as pd
from sklearn.metrics import classification_report
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE = 'iris_v2.csv'

# Get the housing data
df = pd.read_csv(PATH + FILE)

dict_map = {'Iris-setosa':0,'Iris-versicolor':1, 'Iris-virginica':2}
df['target'] = df['iris_type'].map(dict_map)

y = df['target']
X = df.copy()
del X['target']
del X['iris_type']

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head(5))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

ada_boost = AdaBoostClassifier()
grad_boost = GradientBoostingClassifier()
xgb_boost = XGBClassifier()
lr = LogisticRegression(fit_intercept=True, solver='liblinear')

classifiers = [ada_boost, grad_boost, xgb_boost, lr]

for clf in classifiers:
    print(clf.__class__.__name__)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    report = classification_report(y_test, predictions)
    print(report)
