import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

X = np.array(  [[5,3],
                [10,15],
                [15,12],
                [24,10],
                [30,30],
                [85,70],
                [71,80],
                [60,78],
                [70,55],
                [80,91],])

#---------------------------------------------
# Draw data scatter with labels by each point.
#---------------------------------------------
labels = range(1, 11)
plt.scatter(X[:,0],X[:,1])

# Add labels to points.
for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(label, xy=(x, y),
        xytext=(-1, 2), # position of text relative to point.
        textcoords='offset points', ha='right', va='bottom')
plt.show()

#---------------------------------------------
# Draw dendrogram.
#---------------------------------------------
from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(X, 'single')
labelList = range(1, 11)
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)
print(cluster.labels_)

plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()
