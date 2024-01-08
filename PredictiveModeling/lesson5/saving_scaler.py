import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import RobustScaler
from pickle import dump, load

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

# Adding an intercept *** This is requried ***. Don't forget this step.
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc_x = RobustScaler()
X_train_scaled = sc_x.fit_transform(X_train)

# Create y scaler. Only scale y_train since evaluation
# will use the actual size y_test.
sc_y = RobustScaler()
y_train_scaled = sc_y.fit_transform(np.array(y_train).reshape(-1, 1))

# Save the fitted scalers.
dump(sc_x, open('sc_x.pkl', 'wb'))
dump(sc_y, open('sc_y.pkl', 'wb'))

# Build model with training data.
model = sm.OLS(y_train_scaled, X_train_scaled).fit()

# Load the scalers.
loaded_scalerX = load(open('sc_x.pkl', 'rb'))
loaded_scalery = load(open('sc_y.pkl', 'rb'))

X_test_scaled = loaded_scalerX.transform(X_test)
unscaled_predictions = model.predict(X_test_scaled)  # make predictions

# Rescale predictions back to actual size range.
predictions = loaded_scalery.inverse_transform(
    np.array(unscaled_predictions).reshape(-1, 1))

print(model.summary())
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Save model to pickle file.
dump(model, open('model.pkl', 'wb'))

# Load model from pickle file.
loaded_model = load(open('model.pkl', 'rb'))

loaded_model_predictions = loaded_model.predict(X_test_scaled)  # make predictions with loaded model

# Rescale predictions back to actual size range.
loaded_model_predictions = loaded_scalery.inverse_transform(
    np.array(loaded_model_predictions).reshape(-1, 1))

print(loaded_model.summary())
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, loaded_model_predictions)))
