# Load libraries
from sklearn.linear_model    import LogisticRegression
from sklearn                 import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn                 import metrics
import matplotlib.pyplot as plt

import pandas as pd

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target
iris['feature_names']

# Scale data.
minmax = MinMaxScaler()
data_scaled = minmax.fit_transform(X)
data_scaled = pd.DataFrame(data_scaled, columns=['sepal length (cm)', \
            'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
print(data_scaled.head())

# Draw dendrogram.
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.show()

# Predict clusters.
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='manhattan', linkage='complete')
cluster.fit_predict(data_scaled)
print(cluster.labels_)

# Draw scatter.
plt.figure(figsize=(10, 7))
plt.scatter(data_scaled['sepal length (cm)'], data_scaled['sepal width (cm)'], c=cluster.labels_, alpha=0.5)
plt.xlabel("Sepal Length", fontsize=20)
plt.ylabel('Sepal Width', fontsize=20)
plt.show()

# Measure reliability.
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
nmi_result = normalized_mutual_info_score(y, cluster.labels_)
ars_result = adjusted_rand_score(y, cluster.labels_)
print("\n*** Normalized mutual information score: " + str(nmi_result))
print("*** Adjusted rand score: " + str(ars_result))
