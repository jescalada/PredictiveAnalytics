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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn                 import metrics
import statsmodels.api       as sm
import numpy                 as np
from sklearn.metrics import mean_squared_error
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = ROOT_PATH + "\\..\\datasets\\"
CSV_DATA = "winequality.csv"
dataset  = pd.read_csv(PATH + CSV_DATA, sep=',')

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)
print(dataset.head())
print(dataset.describe())
X = dataset[[ 'volatile acidity', 'chlorides', 'total sulfur dioxide',
              'sulphates','alcohol']]

def buildModelAndShowRMSE(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test) # make the predictions by the model
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Adding an intercept *** This is requried ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X = sm.add_constant(X)
y = dataset['quality']

def grid_search(X, y, columnName, trans):
    print("\n**** Column Name: " + columnName)
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
    rmses = {}
    for tran in trans:
        # Create copy to prevent overwrites to reference.
        dfX = X.copy()

        # Transform data.
        dfX['xt'] = trans_func[tran](dfX[columnName])

        del dfX[columnName]  # Drop temporary column.

        X_train, X_test, y_train, y_test = train_test_split(dfX, y, test_size=0.2, random_state=0)

        model_t   = sm.OLS(y_train, X_train).fit()
        predictions_t = model_t.predict(X_test)

        # Calculate RMSE
        mse = mean_squared_error(y_test, predictions_t)
        rmse = np.sqrt(mse)
        rmses[tran] = rmse
    print(rmses)
    return rmses
rmses = grid_search(X, y, "alcohol", ('sqrt', 'neg_inv', 'log', 'exp', 'neg_exp'))
rmses = grid_search(X, y, "volatile acidity", ('sqrt', 'neg_inv', 'log', 'exp', 'neg_exp'))
