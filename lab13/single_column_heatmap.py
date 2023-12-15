import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "\\..\\datasets\\"
CSV_FILE = 'employee_turnover.csv'
df = pd.read_csv(ROOT_PATH + DATASET_DIR + CSV_FILE)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Show transposed data
print(df.head(3).T)

# get the correlation matrix
corr = df.corr()
plt.figure(figsize=(2, 4))
ax = sns.heatmap(corr[['turnover']], linewidth=0.5, vmin=-1, vmax=1, cmap="YlGnBu")
plt.tight_layout()
plt.show()
