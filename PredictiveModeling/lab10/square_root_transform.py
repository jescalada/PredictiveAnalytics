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
x = [0, 20, 50, 100,150,200,250,300,350,400]
y = [0.0, 4.47, 7.07, 10.0, 12.24, 14.14, 15.81, 17.32, 18.70, 20.0]
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
dfX['xt'] = np.sqrt(dfX['x'])
showXandYplot(dfX['xt'] ,y, 'x', 'y=sqrt(x)')

# Build model with transformed x.
model_t       = sm.OLS(y, dfX[['const', 'xt']]).fit()
predictions_t = model_t.predict(dfX[['const', 'xt']])
print(model_t.summary())
showResidualPlotAndRMSE(x,y,predictions_t)
