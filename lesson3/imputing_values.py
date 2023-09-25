import pandas as pd
import numpy  as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import metrics
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"

# Import data into a DataFrame.
file = "babysamp-98.txt"
df = pd.read_table(PATH + file, skiprows=1,
                   delim_whitespace=True,
                   names=('MomAge', 'DadAge', 'MomEduc', 'MomMarital', 'numlive',
                          "dobmm", 'gestation', 'sex', 'weight', 'prenatalstart',
                          'orig.id', 'preemie'))
# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head())        # View a snapshot of the data.
print(df.describe())    # View stats including counts which highlight missing values.


def convert_nan_cells_to_num(col_name: str, dataframe: DataFrame, measure_type: str) -> DataFrame:
    # Create two new column names based on original column name.
    indicator_col_name = 'm_' + col_name  # Tracks whether imputed.
    imputed_col_name = 'imp_' + col_name  # Stores original & imputed data.

    # Get mean or median depending on preference.
    imputed_value = 0
    if measure_type == "median":
        imputed_value = dataframe[col_name].median()
    elif measure_type == "mode":
        imputed_value = float(dataframe[col_name].mode())
    else:
        imputed_value = dataframe[col_name].mean()

    # Populate new columns with data.
    imputed_column = []
    indicator_column = []
    for i in range(len(dataframe)):
        is_imputed = False

        # mi_OriginalName column stores imputed & original data.
        if np.isnan(dataframe.loc[i][col_name]):
            is_imputed = True
            imputed_column.append(imputed_value)
        else:
            imputed_column.append(dataframe.loc[i][col_name])

        # mi_OriginalName column tracks if is imputed (1) or not (0).
        if is_imputed:
            indicator_column.append(1)
        else:
            indicator_column.append(0)

    # Append new columns to dataframe but always keep original column.
    dataframe[indicator_col_name] = indicator_column
    dataframe[imputed_col_name] = imputed_column
    return dataframe


df = convert_nan_cells_to_num('DadAge', df, "median")
df = convert_nan_cells_to_num('MomEduc', df, "median")
df = convert_nan_cells_to_num('prenatalstart', df, "median")
print(df.head(10))

# You can include both the indicator 'm' and imputed 'imp' columns in your model.
# Sometimes both columns boost regression performance and sometimes they do not.
X = df[['gestation',  'm_DadAge', 'imp_DadAge', 'm_prenatalstart', 'imp_prenatalstart',
        'm_MomEduc', 'imp_MomEduc']].values

# Adding an intercept *** This is required ***. Don't forget this step.
X = sm.add_constant(X)
y = df['weight'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build and evaluate model.
model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test) # make the predictions by the model
print(model.summary())
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
