import pandas as pd
import matplotlib.pyplot as plt
import os

# Import data into a DataFrame.
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = f"{ROOT_PATH}\\..\\datasets\\"
# Import data into a DataFrame.

path = f"{DATASET_DIR}\\bodyfat.txt"
df = pd.read_csv(path, skiprows=1,
                   sep='\t',
                   names=('Density', 'Pct.BF', 'Age', 'Weight', 'Height',
                           'Neck', 'Chest', 'Abdomen', 'Waist', 'Hip', 'Thigh',
                          'Ankle', 'Knee', 'Bicep', 'Forearm', 'Wrist'))


# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)
# This line allows us to set the figure size supposedly in inches.
# When rendered in the IDE the output often does not translate to inches.
plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

plt.subplot(2, 2, 1)  # Specfies total rows, columns and image #
# where images are drawn clockwise.
plt.hist(df["Pct.BF"], bins=10)
plt.xlabel("Pct. BF")
plt.ylabel("Frequency")

plt.subplot(2, 2, 2)
plt.hist(df["Age"], bins=10)
plt.xlabel("Age")
plt.ylabel("Frequency")

# 1 is married
# 2 is unmarried
plt.subplot(2, 2, 3)
plt.hist(df["Weight"], bins=10)
plt.xlabel("Weight")
plt.ylabel("Frequency")

# 1 is married. 2 is unmarried.
plt.subplot(2, 2, 4)
plt.hist(df["Height"], bins=10)
plt.xlabel("Height")
plt.ylabel("Frequency")

plt.show()
