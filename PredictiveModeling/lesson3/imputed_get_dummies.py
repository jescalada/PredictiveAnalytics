import pandas as pd
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = f"{ROOT_DIR}\\..\\datasets\\"

path = f"{DATASET_DIR}babysamp-98.txt"

# Import data into a DataFrame.
file_name = "babysamp-98.txt"
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

df = pd.get_dummies(df, columns=['preemie', 'sex'])
print(df.head(10))
