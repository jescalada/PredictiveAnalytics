from statsmodels.formula.api import ols
from sklearn import metrics
import pandas as pd
import numpy as np

x = [0.14, 0.32, 0.35, 0.35, 0.32, 0.21, 0.32, 0.31, 0.47]
y = [0.07, 0.13, 0.33, 0.54, 0.35, 0.15, 0.17, 0.23, 0.43]
df = pd.DataFrame(data={'target': y, 'X': x})

# Generate model to predict target using X.
model = ols('target ~ X', data=df).fit()
print(model.summary())
predictions = model.predict(df['X'])
RMSE = np.sqrt(metrics.mean_squared_error(predictions, y))
print(RMSE)
