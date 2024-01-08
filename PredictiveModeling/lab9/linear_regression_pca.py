import pandas               as pd
import numpy                as np
from sklearn                import model_selection
from sklearn.decomposition  import PCA
from sklearn.linear_model   import LinearRegression
from sklearn.metrics        import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing  import StandardScaler
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "\\..\\datasets\\"
CSV_DATA = "Hitters.csv"

# Drop null values.
df       = pd.read_csv(ROOT_PATH + DATASET_DIR + CSV_DATA).dropna()
df.info()

dummies  = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df.Salary

print("\nSalary stats: ")
print(y.describe())

# Drop the column with the independent variable (Salary),
# and columns for which we created dummy variables.
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')

# Define the feature set X.
X         = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)

# Replace True with 1, and False with 0.
X = X.replace(True, 1, regex=True)
X = X.replace(False, 0, regex=True)

# Calculate and show VIF Scores for original data.
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print("\nOriginal VIF Scores")
print(vif)

# Standardize the data.
X_scaled  = StandardScaler().fit_transform(X)

# Split into training and test sets
X_train, X_test , y_train, y_test = model_selection.train_test_split(X_scaled, y,
                                            test_size=0.25, random_state=1)

# Transform the data using PCA for first 80% of variance.
pca = PCA(.8)
X_reduced_train = pca.fit_transform(X_train)
X_reduced_test  = pca.transform(X_test)

print("\nPrincipal Components")
print(pca.components_)

print("\nExplained variance: ")
print(pca.explained_variance_)

# Train regression model on training data
model = LinearRegression()
model.fit(X_reduced_train, y_train)

# Prediction with test data
pred = model.predict(X_reduced_test)
print()

# Show stats about the regression.
mse = mean_squared_error(y_test, pred)
RMSE = np.sqrt(mse)
print("\nRMSE: " + str(RMSE))

print("\nModel Coefficients")
print(model.coef_)

print("\nModel Intercept")
print(model.intercept_)

from sklearn.metrics import r2_score
print("\nr2_score",r2_score(y_test,pred))

# For each principal component, calculate the VIF and save in dataframe
vif = pd.DataFrame()

# Show the VIF score for the principal components.
print()
vif["VIF Factor"] = [variance_inflation_factor(X_reduced_train, i) for i in range(X_reduced_train.shape[1])]
print(vif)

# Get the eigenvalues and eigenvectors.
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_

# Print the eigenvalues and eigenvectors.
print("\nEigenvalues")
print(eigenvalues)
print("\nEigenvectors")
print(eigenvectors)