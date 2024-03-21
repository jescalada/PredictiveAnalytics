import pandas as pd

# Load training data from train.csv
pd.set_option('display.max_columns', None)
df = pd.read_csv('train.csv')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import os

X = df.copy()
del X['price_range']
y = df['price_range']

# Drop columns that are not important.
del X['blue']
del X['dual_sim']
del X['four_g']
del X['three_g']
del X['touch_screen']
del X['wifi']
del X['clock_speed']
del X['m_dep']
del X['n_cores']
del X['pc']
del X['sc_h']
del X['sc_w']
del X['talk_time']
del X['mobile_wt']
del X['int_memory']
del X['fc']

def get_unfit_models():
    models = list()
    models.append(LogisticRegression())
    models.append(DecisionTreeClassifier())
    models.append(AdaBoostClassifier())
    models.append(RandomForestClassifier(n_estimators=10))
    return models


def evaluate_model(y_test, predictions, model):
    print("\n*** " + model.__class__.__name__)
    report = classification_report(y_test, predictions)
    print(report)
    # Print F1 score.
    f1 = f1_score(y_test, predictions, average='weighted')
    print(f"F1 score: {f1}")


def fit_base_models(X_train, y_train, X_test, models):
    df_predictions = pd.DataFrame()

    # Fit base model and store its predictions in dataframe.
    for i in range(0, len(models)):
        models[i].fit(X_train, y_train)
        predictions = models[i].predict(X_test)
        col_name = str(i)
        df_predictions[col_name] = predictions
    return df_predictions, models


def fit_stacked_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model


# Split data into train, test and validation sets.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.80)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

# Get base models.
unfit_models = get_unfit_models()

# Fit base and stacked models.
df_predictions, models = fit_base_models(X_train, y_train, X_val, unfit_models)
stacked_model = fit_stacked_model(df_predictions, y_val)

# Evaluate base models with validation data.
print("\n** Evaluate Base Models **")
df_validation_predictions = pd.DataFrame()
for i in range(0, len(models)):
    predictions = models[i].predict(X_test)
    col_name = str(i)
    df_validation_predictions[col_name] = predictions
    evaluate_model(y_test, predictions, models[i])

# Evaluate stacked model with validation data.
stacked_predictions = stacked_model.predict(df_validation_predictions)
print("\n** Evaluate Stacked Model **")
evaluate_model(y_test, stacked_predictions, stacked_model)

# Show the dataframe used to fit the stacked model
# Make it reader-friendly by adding labels to the columns.
df_predictions.columns = ['AdaBoost', 'GradientBoost', 'XGBoost', 'LogisticRegression']
print(df_predictions.head())

# Show the model coefficients along with the feature names.
print("\n** Stacked Model Coefficients **")
print(stacked_model.coef_)
print(df_predictions.columns)


# Perform K-Fold Cross Validation

from sklearn.model_selection import cross_val_score
import numpy as np

# Perform 10-fold cross validation
scores = cross_val_score(stacked_model, X_train, y_train, cv=10)
print('Cross-validated scores:', scores)
print('Mean score:', np.mean(scores))
print('Standard deviation of scores:', np.std(scores))
