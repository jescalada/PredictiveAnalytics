import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "\\..\\datasets\\"
PATH = ROOT_PATH + DATASET_DIR
FILE = 'CustomerChurn.csv'

dataframe = pd.read_csv(PATH + FILE)

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

print(dataframe.head())
print(dataframe.describe())

# Plot a boxplot for AccountAge grouped by Churn
dataframe.boxplot(column='AccountAge', by='Churn')
plt.ylim(0, 125)
plt.show()

# Plot a boxplot for MonthlyCharges grouped by Churn
dataframe.boxplot(column='MonthlyCharges', by='Churn')
plt.ylim(0, 20)
plt.show()

# Plot a boxplot for TotalCharges grouped by Churn
dataframe.boxplot(column='TotalCharges', by='Churn')
plt.ylim(0, 2500)
plt.show()

# Compare churn rates according to GenrePreference
y_values = dataframe.groupby('GenrePreference')['Churn'].mean().map(lambda x: x * 100)
dataframe.groupby('GenrePreference')['Churn'].mean().map(lambda x: x * 100).plot(
    kind='bar',
    rot=0,
    title='Churn Rates by Genre Preference',
    xlabel='Genre Preference',
    ylabel='Churn Rate (%)',
    yticks=range(0, 24, 2),
    figsize=(8, 4),
    grid=True
)
for i in range(5):
    plt.text(i, round(y_values[i], 2), round(y_values[i], 2), color='white', ha='center', bbox=dict(facecolor='black'))
plt.show()

# Compare churn rates according to PaymentMethod
y_values = dataframe.groupby('PaymentMethod')['Churn'].mean().map(lambda x: x * 100)
dataframe.groupby('PaymentMethod')['Churn'].mean().map(lambda x: x * 100).plot(
    kind='bar',
    rot=0,
    title='Churn Rates by Payment Method',
    xlabel='Payment Method',
    ylabel='Churn Rate (%)',
    yticks=range(0, 24, 2),
    figsize=(8, 4),
    grid=True
)
for i in range(4):
    plt.text(i, round(y_values[i], 2), round(y_values[i], 2), color='white', ha='center', bbox=dict(facecolor='black'))
plt.show()

# Compare churn rates according to MultiDeviceAccess and DeviceRegistered
y_values = dataframe.groupby(['MultiDeviceAccess', 'DeviceRegistered'])['Churn'].mean().map(lambda x: x * 100)
dataframe.groupby(['MultiDeviceAccess', 'DeviceRegistered'])['Churn'].mean().map(lambda x: x * 100).plot(
    kind='bar',
    rot=0,
    title='Churn Rates by Multi-Device Access and Device Registered',
    xlabel='(Multi-Device, Type of Device)',
    ylabel='Churn Rate (%)',
    yticks=range(0, 24, 2),
    figsize=(12, 4),
    grid=True
)
for i in range(8):
    plt.text(i, round(y_values[i], 2), round(y_values[i], 2), color='white', ha='center', bbox=dict(facecolor='black'))
plt.show()

# Make a scatterplot of MonthlyCharges vs. TotalCharges and color by Churn
# Get a random sample of 1000 data points
# Make the data points small
sample = dataframe.sample(n=10000)
sample.plot(
    kind='scatter',
    x='MonthlyCharges',
    y='TotalCharges',
    c='Churn',
    colormap='winter',
    title='Monthly Charges vs. Total Charges',
    xlabel='Monthly Charges',
    ylabel='Total Charges',
    figsize=(8, 4),
    grid=True,
    s=0.5
)
plt.show()

# Count the missing values in each column
print(dataframe.isnull().sum())

# Replace missing values in AccountAge with the mean
dataframe['AccountAge'].fillna(dataframe['AccountAge'].mean(), inplace=True)

# Bin the AccountAge column into 12 bins of width 10
dataframe['AccountAgeBins'] = pd.cut(dataframe['AccountAge'], bins=12, labels=False)
print(dataframe.head())


# Compare churn rates according to AccountAgeBins
y_values = dataframe.groupby('AccountAgeBins')['Churn'].mean().map(lambda x: x * 100)
dataframe.groupby('AccountAgeBins')['Churn'].mean().map(lambda x: x * 100).plot(
    kind='bar',
    rot=0,
    title='Churn Rates by Account Age in Years',
    xlabel='Account Age (Years)',
    ylabel='Churn Rate (%)',
    yticks=range(0, 34, 2),
    figsize=(10, 4),
    grid=True
)
for i in range(12):
    plt.text(i, round(y_values[i], 2), round(y_values[i], 2), color='white', ha='center', bbox=dict(facecolor='black', alpha=0.6))
plt.show()

# Compare churn rates according to SubscriptionType and ContentType
y_values = dataframe.groupby(['SubscriptionType', 'ContentType'])['Churn'].mean().map(lambda x: x * 100)
dataframe.groupby(['SubscriptionType', 'ContentType'])['Churn'].mean().map(lambda x: x * 100).plot(
    kind='bar',
    rot=0,
    title='Churn Rates by Subscription Type and Content Type',
    xlabel='(Subscription Type, Content Type)',
    ylabel='Churn Rate (%)',
    yticks=range(0, 34, 2),
    figsize=(14, 4),
    fontsize=8,
    grid=True
)
for i in range(9):
    plt.text(i, round(y_values[i], 2), round(y_values[i], 2), color='white', ha='center', bbox=dict(facecolor='black', alpha=0.6))
plt.show()
