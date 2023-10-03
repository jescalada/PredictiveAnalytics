from statsmodels.formula.api import ols
from sklearn import metrics
import pandas as pd
import numpy as np

x = [0.18, 0.33, 0.36, 0.37, 0.4, 0.18, 0.3, 0.34, 0.48]
y = [0.11, 0.16, 0.3, 0.5, 0.43, 0.13, 0.17, 0.27, 0.48]
df = pd.DataFrame(data={'target': y, 'X': x})

# Generate model to predict target using X.
model = ols('target ~ X', data=df).fit()
print(model.summary())
predictions = model.predict(df['X'])
RMSE = np.sqrt(metrics.mean_squared_error(predictions, y))
print(RMSE)
print("")
