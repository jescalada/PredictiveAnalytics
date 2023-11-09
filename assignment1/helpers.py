from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold
import numpy as np
import statsmodels.api as sm



def validate_and_evaluate(X, y, k=10):
    kfold = KFold(n_splits=k, shuffle=True)

    # Create lists to store validation results
    rmse_list = []
    accuracy_list = []
    r2_list = []
    adj_r2_list = []
    aic_list = []
    bic_list = []
    printed_details = False

    best_model = None
    min_rmse = 100000

    # Loop through each fold and create a model each time
    for train_index, test_index in kfold.split(X):
        # Split data into train and test
        X_train = X.iloc[train_index, :]  # Gets all rows with train indexes.
        X_test = X.iloc[test_index, :]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        # Create model
        model = sm.OLS(y_train, X_train).fit()
        predictions = model.predict(X_test)

        # Print details of first fold
        if not printed_details:
            display_model_info(model)
            printed_details = True

        rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

        if rmse < min_rmse:
            min_rmse = rmse
            best_model = model

        # Store results
        accuracy_list.append(metrics.r2_score(y_test, predictions))
        rmse_list.append(rmse)
        r2_list.append(model.rsquared)
        adj_r2_list.append(model.rsquared_adj)
        aic_list.append(model.aic)
        bic_list.append(model.bic)
    print(f"Mean Accuracy: {np.mean(accuracy_list):0.3f} (SD {np.std(accuracy_list):0.3f})")
    print(f"Mean RMSE: {np.mean(rmse_list):0.2f} (SD {np.std(rmse_list):0.2f})")
    print(f"Mean R^2: {np.mean(r2_list):0.3f} (SD {np.std(r2_list):0.3f})")
    print(f"Mean Adjusted R^2: {np.mean(adj_r2_list):0.3f} (SD {np.std(adj_r2_list):0.3f})")
    print(f"Mean AIC: {np.mean(aic_list):0.2f} (SD {np.std(aic_list):0.2f})")
    print(f"Mean BIC: {np.mean(bic_list):0.2f} (SD {np.std(bic_list):0.2f})")
    print(f"Min RMSE: {min_rmse:0.2f}")

    return best_model

def display_model_info(model):
    print(model.summary())
    print(model.params)

def plot_prediction_vs_actual(best_model, x_test, y_test, title):
    predictions = best_model.predict(x_test)
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title('Predicted (Y) vs. Actual (X)' + title)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.tight_layout()
    plt.show()

def plot_residuals_vs_actual(best_model, x_test, y_test, title):
    predictions = best_model.predict(x_test)
    residuals = y_test - predictions
    plt.scatter(y_test, residuals, label='Residuals vs Actual')
    plt.xlabel("Actual")
    plt.ylabel("Residual")
    plt.title('Error Residuals (Y) vs. Actual (X): ' + title)
    plt.plot([y_test.min(), y_test.max()], [0, 0], 'k--')
    plt.tight_layout()
    plt.show()
