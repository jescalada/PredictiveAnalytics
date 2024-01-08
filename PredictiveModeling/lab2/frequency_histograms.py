import pandas as pd
import matplotlib.pyplot as plt
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = f"{ROOT_PATH}\\..\\datasets\\"

# Import data into a DataFrame.
path = f"{DATASET_DIR}\\babysamp-98.txt"
df = pd.read_csv(path, skiprows=1,
                   sep='\t',
                   names=('MomAge', 'DadAge', 'MomEduc', 'MomMarital', 'numlive',
                          "dobmm", 'gestation', 'sex', 'weight', 'prenatalstart',
                          'orig.id', 'preemie'))
# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

plt.hist(df["MomAge"], bins=22)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title('Mother Age')

plt.show()
