from    pandas import DataFrame
import  matplotlib.pyplot as plt
from    sklearn.cluster import KMeans

Data = {
          # 2 2 2 2 2 2 2 2 2 2
    'x': [25, 34, 22, 27, 33, 33, 31, 22, 35, 34, \
          # 0 0 0 0 0 0 0 0 0 0
          67, 54, 57, 43, 50, 57, 59, 52, 65, 47, \
          # 1 1 1 1 1 1 1 1 1 1
          49, 48, 35, 33, 44, 45, 38, 43, 51, 46],

          # 2 2 2 2 2 2 2 2 2 2
    'y': [79, 51, 53, 78, 59, 74, 73, 57, 69, 75, \
          # 0 0 0 0 0 0 0 0 0 0
          51, 32, 40, 47, 53, 36, 35, 58, 59, 50, \
          # 1 1 1 1 1 1 1 1 1 1
          25, 20, 14, 12, 20, 5, 29, 27, 8, 7]}

# Perform clustering.
df         = DataFrame(Data, columns=['x', 'y'])
kmeans     = KMeans(n_clusters=4,  random_state=142).fit(df)
centroids  = kmeans.cluster_centers_

# Show centroids.
print("\n*** centroids ***")
print(centroids)

# Show sample labels.
print("\n*** sample labels ***")
print(kmeans.labels_)

# Parameters: [c for color, s for dot size]
plt.scatter(df['x'], df['y'], c=kmeans.labels_, s=50, alpha=0.5)

# Shows the 3 centroids in red.
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, alpha=0.3)
plt.show()
