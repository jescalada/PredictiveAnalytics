import pandas as pd
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = f"{ROOT_PATH}\\..\\datasets\\"

# Import data into a DataFrame.
path = "babysamp-98.txt"
df = pd.read_table(DATASET_DIR + path, skiprows=1,
                   delim_whitespace=True,
                   names=('MomAge', 'DadAge', 'MomEduc', 'MomMarital', 'numlive',
                          "dobmm", 'gestation', 'sex', 'weight', 'prenatalstart',
                          'orig.id', 'preemie'))
# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head())        # View a snapshot of the data.
print(df.describe())    # View stats including counts which highlight missing values.

df['dadAgeBin'] = pd.cut(x=df['DadAge'], bins=[17, 27, 37, 48])
df['momAgeBin'] = pd.cut(x=df['MomAge'], bins=[13,23,33,42])

temp_df = df[['dadAgeBin', 'momAgeBin', 'sex']] # Isolate columns

# Get dummies
dummy_df = pd.get_dummies(temp_df, columns=['dadAgeBin', 'momAgeBin', 'sex'])
df = pd.concat(([df, dummy_df]), axis=1)  # Join dummy df with original
print(df)
