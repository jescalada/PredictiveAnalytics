import pandas as pd
import os

from pandas import DataFrame

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = f"{ROOT_DIR}\\..\\datasets\\"

path = f"{DATASET_DIR}babysamp-98.txt"

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


def create_binary_dummy_var(dataframe: DataFrame, col_name: str, val0, val1) -> DataFrame:
    """
    Creates a new column with binary values based on the values of an existing column.
    :param dataframe: the dataframe to add the new column to.
    :param col_name: the source column name.
    :param val0: the "False" binary value
    :param val1: the "True" binary value
    :return: a dataframe with the new column added.
    """
    imputed_col_name = "m_" + col_name
    imputed_col = []

    for i in range(len(df)):
        if dataframe.loc[i][col_name] == val0:
            imputed_col.append(0)
        else:
            imputed_col.append(1)
    dataframe[imputed_col_name] = imputed_col
    return dataframe


df = create_binary_dummy_var(df, "preemie", False, True)
df = create_binary_dummy_var(df, "sex", "M", "F")
print(df.head(10))
