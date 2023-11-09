from    sklearn.preprocessing   import StandardScaler
from    sklearn                 import datasets
import  numpy   as np

# Load iris data set and apply standard scaler.
iris         = datasets.load_iris()
X            = iris.data
featureNames = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
y            = iris.target
X_std        = StandardScaler().fit_transform(X)

print(featureNames)

# Generate covariance matrix to show bivariate relationships.
cov_mat = np.cov(np.transpose(X_std))
print('\nCovariance matrix: \n%s' %cov_mat)

# When data is standardized, the covariance matrix is same as the
# correlation matrix.
cor_mat = np.corrcoef(np.transpose(X_std))
print('\nCorrelation matrix: \n%s' %cor_mat)

# Perform an Eigen decomposition on the covariance matrix:
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('\nEigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

import matplotlib.pyplot as plt

# Show the scree plot.
plt.plot([1,2,3,4], eig_vals, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.show()

# Calculate cumulative values.
sumEigenvalues   = eig_vals.sum()
cumulativeValues = []
cumulativeSum    = 0
for i in range(0,len(eig_vals)+1):
    cumulativeValues.append(cumulativeSum)
    if(i<len(eig_vals)):
        cumulativeSum += eig_vals[i] / sumEigenvalues

# Show cumulative variance plot.
import matplotlib.pyplot as plt
plt.plot([0,1,2,3,4], cumulativeValues, 'ro-', linewidth=2)
plt.title('Variance Explained by Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.show()
