import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
from sklearn import metrics
import math


def perform_simple_regression():
    # Initialize collection of X & Y pairs like those used in example 5.
    data = [[0.2, 0.1], [0.32, 0.15], [0.38, 0.4], [0.41, 0.6], [0.43, 0.44]]

    # Create data frame.
    df_sample = pd.DataFrame(data, columns=['X', 'target'])

    # Create training set with 60% of data and test set with 40% of data.
    X_train, X_test, y_train, y_test = train_test_split(
        df_sample['X'], df_sample['target'], train_size=0.6
    )

    # Create DataFrame with test data.
    data_train = {"X": X_train, "target": y_train}
    df_train = pd.DataFrame(data_train, columns=['X', 'target'])

    # Generate model to predict target using X.
    model = ols('target ~ X', data=df_train).fit()
    y_prediction = model.predict(X_test)

    # Present X_test, y_test, y_predict and error sum of squares.
    data = {"X_test": X_test, "y_test": y_test, "y_prediction": y_prediction}
    df_result = pd.DataFrame(data, columns=['X_test', 'y_test', 'y_prediction'])
    df_result['y_test - y_pred'] = (df_result['y_test'] - df_result['y_prediction'])
    df_result['(y_test - y_pred)^2'] = (df_result['y_test'] - df_result['y_prediction']) ** 2

    # Present X_test, y_test, y_predict and error sum of squares.
    print(df_result)

    # Manually calculate the deviation between actual and predicted values.
    rmse = math.sqrt(df_result['(y_test - y_pred)^2'].sum() / len(df_result))
    print("RMSE is average deviation between actual and predicted values: "
          + str(rmse))

    # Show faster way to calculate deviation between actual and predicted values.
    rmse2 = math.sqrt(metrics.mean_squared_error(y_test, y_prediction))
    print("The automated root mean square error calculation is: " + str(rmse2))


perform_simple_regression()
