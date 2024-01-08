import pandas               as pd
import statsmodels.api      as sm
import numpy                as np
import matplotlib.pyplot    as plt
import os
#------------------------------------------------
# Shows plot of x vs. y.
#------------------------------------------------
from sklearn.metrics import mean_squared_error


def showXandYplot(x,y, xtitle, title):
    plt.figure(figsize=(8, 4))
    plt.plot(x,y,color='blue')
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel('y')
    plt.show()

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "\\..\\datasets\\"
PATH = ROOT_PATH + DATASET_DIR
FILE   = 'abs.csv'
df     = pd.read_csv(PATH + FILE)

x = df[['abs(450nm)']]  # absorbance
y = df[['ug/L']]        # protein concentration
showXandYplot(x,y, 'absorbance x', 'Protein Concentration(y) and Absorbance(x)')

# Show raw x and y relationship
x = sm.add_constant(x)

# Show model.
model       = sm.OLS(y, x).fit()
predictions = model.predict(x)
print(model.summary())

# Show RMSE.
preddf      = pd.DataFrame({"predictions":predictions})
residuals   = y['ug/L']-preddf['predictions']
resSq       = [i**2 for i in residuals]
rmse        = np.sqrt(np.sum(resSq)/len(resSq))
print("RMSE: " + str(rmse))

# Show the residual plot
plt.scatter(x['abs(450nm)'],residuals)
plt.show()

def grid_search(dfX, y, trans):
    trans_func = {
        'sqrt': lambda x: np.sqrt(x),
        'inv': lambda x: 1 / x,
        'neg_inv': lambda x: -1 / x,
        'sqr': lambda x: x * x,
        'log': lambda x: np.log(x),
        'neg_log': lambda x: -np.log(x),
        'exp': lambda x: np.exp(x),
        'neg_exp': lambda x: np.exp(-x),
    }
    dfTransformations = pd.DataFrame()
    for tran in trans:
        # Transform x
        dfX['xt'] = trans_func[tran](dfX['x'])
        model_t = sm.OLS(y, dfX[['const', 'xt']]).fit()
        predictions_t = model_t.predict(dfX[['const', 'xt']])
        # Calculate RMSE
        mse = mean_squared_error(y, predictions_t)
        rmse = np.sqrt(mse)
        dfTransformations = dfTransformations._append({
            "tran":tran, "rmse":rmse}, ignore_index=True)
    dfTransformations = dfTransformations.sort_values(by=['rmse'])
    return dfTransformations

# Find the best transformation
dfX = pd.DataFrame({"x": x['abs(450nm)']})
dfX = sm.add_constant(dfX)

rmse_df = grid_search(dfX, y, ['sqrt', 'inv', 'neg_inv', 'sqr', 'log', 'neg_log', 'exp', 'neg_exp'])
print(rmse_df)

# Build a model with exponential transformation
dfX['xt'] = np.exp(dfX['x'])
model_t = sm.OLS(y, dfX[['const', 'xt']]).fit()
predictions_t = model_t.predict(dfX[['const', 'xt']])
print(model_t.summary())

# Show RMSE.
preddf = pd.DataFrame({"predictions":predictions_t})
residuals = y['ug/L']-preddf['predictions']
resSq = [i**2 for i in residuals]
rmse = np.sqrt(np.sum(resSq)/len(resSq))
print("RMSE: " + str(rmse))

# Show the residual plot
plt.scatter(dfX['xt'],residuals)
plt.show()
