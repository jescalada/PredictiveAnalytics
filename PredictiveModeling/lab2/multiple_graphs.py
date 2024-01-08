import pandas as pd
import matplotlib.pyplot as plt
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

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)
# This line allows us to set the figure size supposedly in inches.
# When rendered in the IDE the output often does not translate to inches.
plt.subplots(nrows=2, ncols=3, figsize=(14, 7))

plt.subplot(2, 3, 1)  # Specfies total rows, columns and image #
# where images are drawn clockwise.
plt.hist(df["MomAge"], bins=22)
plt.xlabel("Mom Age")
plt.ylabel("Frequency")
# plt.title('Mother Age')

plt.subplot(2, 3, 2)
plt.hist(df["MomEduc"], bins=10)
plt.xlabel("Mom Education")
plt.ylabel("Frequency")

# 1 is married
# 2 is unmarried
plt.subplot(2, 3, 3)
plt.hist(df["MomMarital"], bins=10)
t11 = ['', 'Y', 'N']
plt.xticks(range(len(t11)), t11, rotation=50)

# plt.xticks(['Mar', '2'], rotation=50)
plt.xlabel("Mom Married")
plt.ylabel("Frequency")

# 1 is married. 2 is unmarried.
plt.subplot(2, 3, 4)
plt.hist(df["numlive"], bins=10)
plt.xlabel("# Kids")
plt.ylabel("Frequency")

plt.subplot(2, 3, 5)  # of rows, # of columns, # plots.
plt.hist(df["DadAge"], bins=22)
plt.xlabel("Dad Age")
plt.ylabel("Frequency")

plt.subplot(2, 3, 6)
plt.hist(df["dobmm"], bins=12)
plt.xlabel("Birth Month")
plt.ylabel("Frequency")

plt.show()
