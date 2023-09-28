import pandas as pd
import matplotlib.pyplot as plt
import os

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Import data into a DataFrame.
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = f"{ROOT_PATH}\\..\\datasets\\"
file = "bodyfat.txt"
df = pd.read_csv(DATASET_DIR + file, sep='\t')
plt.rcParams.update({'font.size': 14})

print(df.head())
plt.subplots(nrows=1, ncols=4,  figsize=(14, 7))

plt.subplot(1, 4, 1)
boxplot = df.boxplot(column=['Pct.BF'])

plt.subplot(1, 4, 2)
boxplot = df.boxplot(column=['Age'])

plt.subplot(1, 4, 3)
boxplot = df.boxplot(column=['Weight'])

plt.subplot(1, 4, 4)
boxplot = df.boxplot(column=['Height'])

plt.show()
