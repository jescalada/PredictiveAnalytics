from pandas import DataFrame
import math

import numpy as np

# Centroids and labels are already known so assign them manually.
centroids = np.zeros((3, 2))
centroids[0][0] = 55.1
centroids[0][1] = 46.1
centroids[1][0] = 43.2
centroids[1][1] = 16.7
centroids[2][0] = 29.6
centroids[2][1] = 66.8

clusterLabels   = [0,1,2]

# Receives X_test values and finds closest cluster for each.
def predictCluster(centroids, X_test, clusterLabels):
    bestClusterList = []

    for i in range(0, len(X_test)):
        smallestDistance = None
        bestCluster      = None

        # Compare each X value proximity with all centroids.
        for row in range(centroids.shape[0]):
            distance = 0

            # Get absolute distance between centroid and X.
            for col in range(centroids.shape[1]):
                distance += \
                math.sqrt((centroids[row][col] - X_test.iloc[i][col])**2)

            # Initialize bestCluster and smallestDistance during first iteration.
            # OR re-assign bestCluster if smaller distance to centroid found.
            if(bestCluster == None or distance < smallestDistance):
                bestCluster      = clusterLabels[row]
                smallestDistance = distance

        bestClusterList.append(bestCluster)

    return bestClusterList

X_test          = DataFrame({"x":[55],"y":[35]})
predictions     = predictCluster(centroids, X_test, clusterLabels)
print(predictions)
