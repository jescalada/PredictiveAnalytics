import pickle

import numpy as np
import pandas as pd
import os
from constants import *


def predict(dataframe):
    rng = np.random.default_rng()

    # Iterate through each row and impute missing values
    for index, row in dataframe.iterrows():
        if pd.isnull(row['AccountAge']):
            if row['Churn'] == 1:
                dataframe.at[index, 'AccountAge'] = rng.normal(MEAN_ACCOUNT_AGE_CHURNED, STD_ACCOUNT_AGE_CHURNED)
            else:
                dataframe.at[index, 'AccountAge'] = rng.normal(MEAN_ACCOUNT_AGE_NOT_CHURNED,
                                                                    STD_ACCOUNT_AGE_NOT_CHURNED)
        if pd.isnull(row['ViewingHoursPerWeek']):
            if row['Churn'] == 1:
                dataframe.at[index, 'ViewingHoursPerWeek'] = rng.normal(MEAN_VIEWING_HOURS_PER_WEEK_CHURNED,
                                                                        STD_VIEWING_HOURS_PER_WEEK_CHURNED)
            else:
                dataframe.at[index, 'ViewingHoursPerWeek'] = rng.normal(MEAN_VIEWING_HOURS_PER_WEEK_NOT_CHURNED,
                                                                        STD_VIEWING_HOURS_PER_WEEK_NOT_CHURNED)
        if pd.isnull(row['AverageViewingDuration']):
            if row['Churn'] == 1:
                dataframe.at[index, 'AverageViewingDuration'] = rng.normal(MEAN_AVERAGE_VIEWING_DURATION_CHURNED,
                                                                           STD_AVERAGE_VIEWING_DURATION_CHURNED)
            else:
                dataframe.at[index, 'AverageViewingDuration'] = rng.normal(MEAN_AVERAGE_VIEWING_DURATION_NOT_CHURNED,
                                                                           STD_AVERAGE_VIEWING_DURATION_NOT_CHURNED)

    # Bin the AccountAge column into 10 bins of width 10
    print("Attempting to bin AccountAge for test_dataframe:")
    print(f"test_dataframe head: \n{dataframe.head()}")
    print(f"test_dataframe AccountAge head: \n{dataframe['AccountAge'].head()}")
    dataframe['AccountAgeBins'] = pd.cut(dataframe['AccountAge'], bins=10, labels=False)

    # Round user ratings to nearest 0.1
    dataframe['UserRating'] = dataframe['UserRating'].round(1)

    # Bin the UserRating column into 4 bins of width 1
    dataframe['UserRatingBins'] = pd.cut(dataframe['UserRating'], bins=4)

    # Load scaler from scaler.pkl
    scaler = pickle.load(open(ROOT_PATH + "\\scaler.pkl", 'rb'))
    dataframe[['AccountAge', 'MonthlyCharges', 'ViewingHoursPerWeek', 'AverageViewingDuration',
                    'ContentDownloadsPerMonth', 'SupportTicketsPerMonth']] = scaler.fit_transform(
        dataframe[['AccountAge', 'MonthlyCharges', 'ViewingHoursPerWeek', 'AverageViewingDuration',
                        'ContentDownloadsPerMonth', 'SupportTicketsPerMonth']])
    print("After scaling:")
    print(dataframe.head())

    # Add dummies for categorical columns
    dataframe = pd.get_dummies(dataframe,
                                    columns=['SubscriptionType', 'PaymentMethod', 'PaperlessBilling', 'ContentType',
                                             'MultiDeviceAccess', 'DeviceRegistered', 'GenrePreference', 'Gender',
                                             'ParentalControl', 'SubtitlesEnabled',
                                             'AccountAgeBins', 'UserRatingBins'])

    # Keep only the columns found through RFE:
    dataframe_x = dataframe[['AccountAge',
                                       'MonthlyCharges',
                                       'ViewingHoursPerWeek',
                                       'AverageViewingDuration',
                                       'ContentDownloadsPerMonth',
                                       'SupportTicketsPerMonth',
                                       'PaymentMethod_Mailed check',
                                       'PaperlessBilling_No',
                                       'ContentType_TV Shows',
                                       'MultiDeviceAccess_No']]

    # Load the model from model.pkl
    model = pickle.load(open(ROOT_PATH + "\\model.pkl", 'rb'))

    # Predict the values
    y_pred = model.predict(dataframe_x)

    # Return the predictions in a dataframe only with Churn column
    return pd.DataFrame(y_pred, columns=['Churn'])


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
FILE = "\\CustomerChurn_Mystery.csv"
X = pd.read_csv(ROOT_PATH + FILE)
print(X.head())

predictions = predict(X)
print(predictions.head())

# Write the predictions to CustomerChurn_Predictions.csv line by line
with open(ROOT_PATH + "\\CustomerChurn_Predictions.csv", 'w') as f:
    f.write("Churn\n")
    for index, row in predictions.iterrows():
        f.write(str(row['Churn']) + "\n")

