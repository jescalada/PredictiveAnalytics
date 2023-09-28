import os
import pandas as pd
import numpy as np
from scipy import stats

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = f"{ROOT_PATH}\\..\\datasets\\"
file = "babysamp-98.txt"

# Import data into a DataFrame.
df = pd.read_csv(DATASET_DIR + file, sep='\t')

df_sub = df[['MomAge', 'MomMarital', 'numlive', 'dobmm', 'gestation']]

z = np.abs(stats.zscore(df_sub))
print(z)

Z_THRESHOLD = 3
print(np.where(z > Z_THRESHOLD))

print(df_sub.loc[48][[2]])
print(df_sub.loc[82][[4]])
