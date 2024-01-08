import pandas as pd
import matplotlib.pyplot as plt
import os

# Import data into a DataFrame.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = f"{ROOT_DIR}\\..\\datasets\\"

file = "babysamp-98.txt"
df = pd.read_csv(DATASET_DIR + file, sep='\t')

# Rename the columns so they are more reader-friendly.
df = df.rename({'MomAge': 'Mom Age', 'DadAge': 'Dad Age',
                'MomEduc': 'Mom Edu', 'weight': 'Weight'}, axis=1)  # new method

# This line allows us to set the figure size supposedly in inches.
# When rendered in the IDE the output often does not translate to inches.
plt.subplots(nrows=1, ncols=3,  figsize=(14, 7))

plt.subplot(1, 3, 1)  # Specifies total rows, columns and image #
                      # where images are drawn clockwise.
plt.xticks([], ())
boxplot = df.boxplot(column=['Mom Age', 'Dad Age'])

plt.subplot(1, 3, 2)  # Specifies total rows, columns and image #
                      # where images are drawn clockwise.
boxplot = df.boxplot(column=['Mom Edu'])

plt.subplot(1, 3, 3)  # Specifies total rows, columns and image #
                      # where images are drawn clockwise.
boxplot = df.boxplot(column=['Weight'])

plt.show()
