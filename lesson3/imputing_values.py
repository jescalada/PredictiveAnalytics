import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split
import statsmodels.api       as     sm
from   sklearn               import metrics

# Import data into a DataFrame.
path = "/Users/pm/Desktop/DayDocs/PythonForDataAnalytics/workingData/babysamp-98.txt"
df = pd.read_table(path, skiprows=1,
                   delim_whitespace=True,
                   names=('MomAge', 'DadAge', 'MomEduc', 'MomMarital', 'numlive',
                          "dobmm", 'gestation', 'sex', 'weight', 'prenatalstart',
                          'orig.id', 'preemie'))
# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head())        # View a snapshot of the data.
print(df.describe())    # View stats including counts which highlight missing values.

def convertNAcellsToNum(colName, df, measureType):
    # Create two new column names based on original column name.
    indicatorColName = 'm_'   + colName # Tracks whether imputed.
    imputedColName   = 'imp_' + colName # Stores original & imputed data.

    # Get mean or median depending on preference.
    imputedValue = 0
    if(measureType=="median"):
        imputedValue = df[colName].median()
    elif(measureType=="mode"):
        imputedValue = float(df[colName].mode())
    else:
        imputedValue = df[colName].mean()

    # Populate new columns with data.
    imputedColumn  = []
    indictorColumn = []
    for i in range(len(df)):
        isImputed = False

        # mi_OriginalName column stores imputed & original data.
        if(np.isnan(df.loc[i][colName])):
            isImputed = True
            imputedColumn.append(imputedValue)
        else:
            imputedColumn.append(df.loc[i][colName])

        # mi_OriginalName column tracks if is imputed (1) or not (0).
        if(isImputed):
            indictorColumn.append(1)
        else:
            indictorColumn.append(0)

    # Append new columns to dataframe but always keep original column.
    df[indicatorColName] = indictorColumn
    df[imputedColName]   = imputedColumn
    return df

df = convertNAcellsToNum('DadAge', df, "mode")
df = convertNAcellsToNum('MomEduc', df, "mean")
df = convertNAcellsToNum('prenatalstart', df, "median")
print(df.head(10))

# You can include both the indicator 'm' and imputed 'imp' columns in your model.
# Sometimes both columns boost regression performance and sometimes they do not.
X = df[[ 'gestation',  'm_DadAge', 'imp_DadAge', 'm_prenatalstart', 'imp_prenatalstart',
         'm_MomEduc', 'imp_MomEduc']].values

# Adding an intercept *** This is requried ***. Don't forget this step.
X = sm.add_constant(X)
y = df['weight'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build and evaluate model.
model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test) # make the predictions by the model
print(model.summary())
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
