import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import os

# Import data into a DataFrame.
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = f"{ROOT_PATH}\\..\\datasets\\"
path = f"{DATASET_DIR}\\bodyfat.txt"

df = pd.read_csv(path, skiprows=1,
                   sep='\t',
                   names=('Density', 'Pct.BF', 'Age',   'Weight', 'Height',
                          'Neck', 'Chest','Abdomen', 'Waist', 'Hip', 'Thigh',
                          'Ankle', 'Knee', 'Bicep',
                          'Forearm',
                          'Wrist'))
scatter_matrix(df, figsize=(12,12))
plt.show()
