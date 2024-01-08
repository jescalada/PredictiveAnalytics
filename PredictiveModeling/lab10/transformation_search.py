import matplotlib.pyplot    as plt
from sklearn.metrics import mean_squared_error
import pandas               as pd
import statsmodels.api      as sm
import numpy                as np

#------------------------------------------------
# Shows plot of x vs. y.
#------------------------------------------------
def showXandYplot(x,y, xtitle, title):
    plt.figure(figsize=(8, 4))
    plt.plot(x,y,color='blue')
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel('y')
    plt.show()

#------------------------------------------------
# Shows plot of actual vs. predicted and RMSE.
#------------------------------------------------
def showResidualPlotAndRMSE(x, y, predictions):
    xmax      = max(x)
    xmin      = min(x)
    residuals = y - predictions

    plt.figure(figsize=(8, 3))
    plt.title('x and y')
    plt.plot([xmin,xmax],[0,0],'--',color='black')
    plt.title("Residuals")
    plt.scatter(x,residuals,color='red')
    plt.show()

    # Calculate RMSE
    mse = mean_squared_error(y,predictions)
    rmse = np.sqrt(mse)
    print("RMSE: " + str(rmse))

# Section A: Define the raw data.
x = [0.1, 0.8, .99, 1.4, 2, 2.1, 2.4, 3.8]
y = [-2.303, -0.2231, -0.010050, 0.3364, 0.6931,
      0.74194, 0.87547, 1.3350]

showXandYplot(x,y, 'x', 'x and y')

# Show raw x and y relationship
dfX = pd.DataFrame({"x": x})
dfY = pd.DataFrame({"y": y})
dfX = sm.add_constant(dfX)

# Show residuals from y(x)
model       = sm.OLS(y, dfX).fit()
predictions = model.predict(dfX)
print(model.summary())
showResidualPlotAndRMSE(x,y,predictions)

x = [0.1, 0.8, .99, 1.4, 2, 2.1, 2.4, 3.8]
y = pd.Series([-2.303, -0.2231, -0.010050, 0.3364, 0.6931, 0.74194, 0.87547,
               1.3350])
# Show raw x and y relationship
dfX = pd.DataFrame({"x": x})
dfX = sm.add_constant(dfX)
# Pass a tuple of strings to test for trans. Refer to trans_func for the
# keywords.
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
rmsedf = grid_search(dfX, y, ('sqrt', 'neg_inv', 'log', 'exp', 'neg_exp'))
print(rmsedf)
