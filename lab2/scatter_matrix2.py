import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import os

# Import data into a DataFrame.
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = f"{ROOT_PATH}\\..\\datasets\\"

path = f"{DATASET_DIR}\\babysamp-98.txt"
df = pd.read_csv(path, skiprows=1,
                 sep='\t',
                 names=('MomAge', 'DadAge', 'MomEduc', 'MomMarital', 'numlive',
                        "dobmm", 'gestation', 'sex', 'weight', 'prenatalstart',
                        'orig.id', 'preemie'))
sub_df = df[['MomAge', 'gestation', 'weight']]

scatter_matrix(sub_df, figsize=(12,12))
plt.show()
