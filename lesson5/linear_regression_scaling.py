import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import RobustScaler

wine = datasets.load_wine()
dataset = pd.DataFrame(
    data=np.c_[wine['data'], wine['target']],
    columns=wine['feature_names'] + ['target']
)

# Create copy to prevent overwrite.
X = dataset.copy()
del X['target']  # Remove target variable
del X['hue']  # Remove unwanted features
del X['ash']
del X['magnesium']
del X['malic_acid']
del X['alcohol']

y = dataset['target']

# Adding an intercept *** This is required ***. Don't forget this step.
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
sc_x = RobustScaler()
X_train_scaled = sc_x.fit_transform(X_train)
X_test_scaled = sc_x.transform(X_test)

# Create y scaler. Only scale y_train since evaluation
# will use the actual size y_test.
sc_y = RobustScaler()
y_train_scaled = sc_y.fit_transform(np.array(y_train).reshape(-1, 1))

model = sm.OLS(y_train_scaled, X_train_scaled).fit()
unscaled_predictions = model.predict(X_test_scaled)  # make predictions

# Rescale predictions back to actual size range.
predictions = sc_y.inverse_transform(np.array(unscaled_predictions).reshape(-1, 1))

print(model.summary())
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
