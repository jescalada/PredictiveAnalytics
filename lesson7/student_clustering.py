from    pandas import DataFrame
import pandas as pd
import  matplotlib.pyplot as plt
from    sklearn.cluster import KMeans
import os

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = f"{ROOT_PATH}\\..\\datasets\\"
data = pd.read_csv(DATASET_DIR + "individualTeammatePreferences.csv")

df = data.copy()
del df['Full_Name']
del df['Unnamed: 0']
print(df)

kmeans     = KMeans(n_clusters=25, random_state=0).fit(df)
centroids  = kmeans.cluster_centers_

# Show centroids.
print("\n*** centroids ***")
print(centroids)

# Show sample labels.
print("\n*** sample labels ***")
print(kmeans.labels_)

data['Label'] = 0.0
colPosition = len(data.keys())-1
for i in range(len(data)):
    data.iat[i,colPosition] = kmeans.labels_[i]

data = data.sort_values(['Label'])
pd.set_option('display.max_rows', data.shape[0]+1)
print(data.head(len(data)))
