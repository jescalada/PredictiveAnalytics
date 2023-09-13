import matplotlib.pylab as plt
import os
import pandas as pd
import seaborn as sns

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = f"{ROOT_PATH}\\..\\datasets\\"

# Import data into a DataFrame.
path = f"{DATASET_DIR}bodyfat.txt"
df = pd.read_csv(path, skiprows=1,
                   sep='\t',
                   names=('Density', 'Pct.BF', 'Age',   'Weight', 'Height',
                          'Neck', 'Chest','Abdomen', 'Waist', 'Hip', 'Thigh',
                          'Ankle', 'Knee', 'Bicep',
                          'Forearm',
                          'Wrist'))
# Compute the correlation matrix
corr = df.corr()

# plot the heatmap

sns.set(rc = {'figure.figsize':(6,4)})
sns.heatmap(corr[['Weight']],
            linewidths=0.1, vmin=-1, vmax=1,
            cmap="YlGnBu")
plt.show()
