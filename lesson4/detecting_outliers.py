import pandas as pd
import os
from scipy import stats
import numpy as np

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "USA_Housing.csv"
dataset = pd.read_csv(DATASET_DIR + CSV_DATA)

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

dfSub = dataset[['Avg. Area Income', 'Avg. Area House Age',
                 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', "Area Population",
                 'Price']]

z = np.abs(stats.zscore(dfSub))
print(z)

THRESHOLD = 3
print(np.where(z > THRESHOLD))
print(dfSub.loc[39][[0]])
print(dfSub.loc[4855][[0]])
print(dfSub.loc[228][[4]])
