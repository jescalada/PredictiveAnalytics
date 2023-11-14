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
x = [0.01, 0.2, 0.5, 0.7, 0.9,1,2,3,4,5,6,10,11,12,13,14,15,16,17,18,19,20]
y = [0.99, 0.819, 0.6065, 0.4966, 0.40657, 0.368, 0.1353, 0.0498,
     0.01831, 0.00674, 0.0025, 4.5399e-05, 1.670e-05, 6.1e-06,
     2.260e-06, 8.3153e-07, 3.0590e-07, 1.12e-07, 4.14e-08,
     1.52e-08, 5.60e-09, 2.061e-09]
print(y)

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

# Section B: Transform and plot graph with transformed x.
dfX['xt'] = np.exp(-dfX['x'])
showXandYplot(dfX['xt'] ,y, 'x', 'y=sqrt(x)')

# Build model with transformed x.
model_t       = sm.OLS(y, dfX[['const', 'xt']]).fit()
predictions_t = model_t.predict(dfX[['const', 'xt']])
print(model_t.summary())
showResidualPlotAndRMSE(x,y,predictions_t)
