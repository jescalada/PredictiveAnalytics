import pandas as pd
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Import data into a DataFrame.
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = f"{ROOT_PATH}\\..\\datasets\\"
path = "babysamp-98.txt"
df   = pd.read_table(DATASET_DIR + path, sep='\t')
numericColumns = ['MomAge','DadAge','MomEduc','MomMarital','numlive',
                    'dobmm','gestation','weight','prenatalstart']

dfNumeric = df[numericColumns]

# Show data types for each columns.
print("\n*** Before imputing")
print(df.describe())
print(dfNumeric.head(11))

# Show summaries for objects like dates and strings.
from sklearn.impute import KNNImputer
imputer   = KNNImputer(n_neighbors=5)
dfNumeric = pd.DataFrame(imputer.fit_transform(dfNumeric),
                         columns = dfNumeric.columns)

# Show data types for each columns.
print("\n*** After imputing")
print(dfNumeric.describe())
print(dfNumeric.head(11))
