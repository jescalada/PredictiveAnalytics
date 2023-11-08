from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import pandas as pd

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target
iris['feature_names']

print(iris)

# Scale data.
minmax = MinMaxScaler()
data_scaled = minmax.fit_transform(X)
data_scaled = pd.DataFrame(data_scaled, columns=['sepal length (cm)', \
                                                 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
print(data_scaled.head())

# Perform clustering.
df = DataFrame(data_scaled, columns=['sepal length (cm)', \
                                     'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
kmeans = KMeans(n_clusters=4, random_state=142).fit(df)
centroids = kmeans.cluster_centers_

# Show centroids.
print("\n*** centroids ***")
print(centroids)

# Show sample labels.
print("\n*** sample labels ***")
print(kmeans.labels_)

# Parameters: [c for color, s for dot size]
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=kmeans.labels_, s=50, alpha=0.5)

# Shows the centroids in red.
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, alpha=0.3)
plt.show()

# Print the NMI and ARS
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

nmi_result = normalized_mutual_info_score(y, kmeans.labels_)
ars_result = adjusted_rand_score(y, kmeans.labels_)

print("\n*** Normalized mutual information score: " + str(nmi_result))
print("*** Adjusted rand score: " + str(ars_result))