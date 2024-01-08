import matplotlib.pylab as plt
import os
import pandas as pd
import seaborn as sns

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = f"{ROOT_PATH}\\..\\datasets\\"

# Import data into a DataFrame.
path = f"{DATASET_DIR}\\babysamp-98.txt"
df = pd.read_csv(path, skiprows=1,
                 sep='\t',
                 names=('MomAge', 'DadAge', 'MomEduc', 'MomMarital', 'numlive',
                        "dobmm", 'gestation', 'sex', 'weight', 'prenatalstart',
                        'orig.id', 'preemie'))
sub_df = df[['MomAge', 'gestation', 'weight']]
# Compute the correlation matrix
corr = sub_df.corr()

# plot the heatmap

sns.set(rc = {'figure.figsize':(6,4)})
sns.heatmap(corr[['weight']],
            linewidths=0.1, vmin=-1, vmax=1,
            cmap="YlGnBu")
plt.show()
