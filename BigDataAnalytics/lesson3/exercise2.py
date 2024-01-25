import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

dta = sm.datasets.sunspots.load_pandas().data
print(dta)
plt.plot(dta['YEAR'], dta['SUNACTIVITY'])
plt.show()

# Show autocorrelation function.
# General correlation of lags with past lags.
plot_acf(dta['SUNACTIVITY'], lags=50)
plt.show()

# Show partial-autocorrelation function.
# Shows correlation of 1st lag with past lags.
plot_pacf(dta['SUNACTIVITY'], lags=50)
plt.show()

print(dta)

# Create back-shifted columns for an attribute.
def add_back_shifted_columns(df, col_name, time_lags):
    for i in range(1, time_lags + 1):
        new_col_name = col_name + "_t-" + str(i)
        df[new_col_name] = df[col_name].shift(i)
    return df

# Generate back-shifted columns for all attributes.
columns = ['SUNACTIVITY']
model_df = dta.copy()
NUM_TIME_STEPS = 12
for i in range(0, len(columns)):
    model_df = add_back_shifted_columns(model_df, columns[i],
                                        NUM_TIME_STEPS)

# Drop rows with NaN values.
model_df = model_df.dropna()
print(model_df)

# Get test and train data (use only last ten samples for test).
train_df = model_df[:-10]
test_df = model_df[-11:]

# Build model.
y = train_df[['SUNACTIVITY']]
X = train_df[['SUNACTIVITY_t-1', 'SUNACTIVITY_t-2', 'SUNACTIVITY_t-9']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Make predictions.
test_df['PREDICTION'] = model.predict(sm.add_constant(test_df[['SUNACTIVITY_t-1',
                                                                'SUNACTIVITY_t-2',
                                                                'SUNACTIVITY_t-9']]))
print(test_df)

# Plot raw train, test and predictions.
train_df['SUNACTIVITY'].plot(legend=True, label='TRAIN')
test_df['SUNACTIVITY'].plot(legend=True, label='TEST', figsize=(6, 4))
test_df['PREDICTION'].plot(legend=True, label='PREDICTION')
plt.show()

# Show model summary.
print(model.summary())

# Show RMSE
rmse = sqrt(mean_squared_error(test_df['SUNACTIVITY'], test_df['PREDICTION']))
print(rmse)
