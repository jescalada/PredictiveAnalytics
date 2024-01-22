from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Prepare the data.
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "Social_Network_Ads.csv"

df = pd.read_csv(PATH + CSV_DATA)
df = pd.get_dummies(df, columns=['Gender'])
del df['User ID']

X = df.copy()
del X['Purchased']
y = df['Purchased']


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
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70)
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
