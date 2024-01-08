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
y = [4.60, 1.60, 0.69, 0.35, 0.11, 0.0, -0.69, -1.1, -1.4, -1.61, -1.79, -2.3, -2.4,
    -2.49, -2.57, -2.64, -2.71, -2.77, -2.83, -2.89, -2.94, -2.996]
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
dfX['xt'] = -np.log(dfX['x'])
showXandYplot(dfX['xt'] ,y, 'x', 'y=-log(x)')

# Build model with transformed x.
model_t       = sm.OLS(y, dfX[['const', 'xt']]).fit()
predictions_t = model_t.predict(dfX[['const', 'xt']])
print(model_t.summary())
showResidualPlotAndRMSE(x,y,predictions_t)
