import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT_DATA = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "\\..\\datasets\\"
FILE = 'loan_v2.csv'

# Visualize the data.
dataframe = pd.read_csv(ROOT_DATA + DATASET_DIR + FILE)
print(dataframe.head(10))
print(dataframe.describe())

# Get only the numeric columns.
numeric_columns = dataframe.select_dtypes(include=['float64', 'int64'])

# Get the correlation matrix and plot a heatmap.
corr = numeric_columns.corr()
sns.set(rc={'figure.figsize': (6, 4)})

# Extend the heatmap to show all the labels
sns.heatmap(corr[['Loan Sanction Amount (USD)']],
            linewidths=0.1, vmin=-1, vmax=1,
            cmap="YlGnBu")
plt.tight_layout()
plt.show()

# Show a histogram of the Loan Sanction Amount (USD) column.
plt.hist(dataframe['Loan Sanction Amount (USD)'], bins=40, range=(0, 200000), color='red')
plt.xlabel("Loan Sanction Amount (USD)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Get the mean, median and SD
print("Mean Sanction Amount: ", dataframe['Loan Sanction Amount (USD)'].mean())
print("Median Sanction Amount: ", dataframe['Loan Sanction Amount (USD)'].median())
print("SD Sanction Amount: ", dataframe['Loan Sanction Amount (USD)'].std())

# Show a histogram of the Age column.
plt.hist(dataframe['Age'], bins=65, range=(0, 65))
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Get the mean, median and SD
print("Mean Age: ", dataframe['Age'].mean())
print("Median Age: ", dataframe['Age'].median())
print("SD Age: ", dataframe['Age'].std())

# Show a histogram of the Income (USD) column.
plt.hist(dataframe['Income (USD)'], bins=40, range=(0, 10000))
plt.xlabel("Income (USD)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Get the mean, median and SD
print("Mean Income: ", dataframe['Income (USD)'].mean())
print("Median Income: ", dataframe['Income (USD)'].median())
print("SD Income: ", dataframe['Income (USD)'].std())

# Remove some top incomes and recompute the mean, median and SD
dataframe_no_income_outliers = dataframe[dataframe['Income (USD)'] < 20000]

new_income_mean = dataframe_no_income_outliers['Income (USD)'].mean()
new_income_sd = dataframe_no_income_outliers['Income (USD)'].std()
new_income_median = dataframe_no_income_outliers['Income (USD)'].median()

# Print the new mean, median and SD
print("New Mean Income: ", new_income_mean)
print("New Median Income: ", new_income_median)
print("New SD Income: ", new_income_sd)

# Show a histogram of the Property Price column. Show scale for x axis in dollars
plt.hist(dataframe['Property Price'], bins=40, range=(0, 500000), color='green')
plt.xlabel("Property Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Get the mean, median and SD
print("Mean Property Price: ", dataframe['Property Price'].mean())
print("Median Property Price: ", dataframe['Property Price'].median())
print("SD Property Price: ", dataframe['Property Price'].std())

# Make a scatterplot of the Credit Score and Loan Sanction Amount.
# Make the dots small
plt.scatter(dataframe['Credit Score'], dataframe['Loan Sanction Amount (USD)'], color='red', s=0.2)
plt.xlabel("Credit Score")
plt.ylabel("Loan Sanction Amount (USD)")
plt.tight_layout()
plt.show()


# slope, intercept = np.polyfit(dataframe['Credit Score'], dataframe['Loan Sanction Amount (USD)'], deg=1)
plt.scatter(dataframe_no_income_outliers['Credit Score'], dataframe_no_income_outliers['Loan Sanction Amount (USD)'], color='blue', s=0.2)
plt.xlabel("Credit Score")
plt.ylabel("Loan Sanction Amount (USD)")

# Plot the best fit line
x = np.array([550, 900])
# y = slope * x + intercept
# plt.plot(x, y, color='red')
plt.tight_layout()
plt.show()

# Make best fit line for the Income and Loan Sanction Amount and plot it along with the scatterplot
# Get the slope and intercept of the best fit line
slope, intercept = np.polyfit(dataframe_no_income_outliers['Income (USD)'], dataframe_no_income_outliers['Loan Sanction Amount (USD)'], 1)

# Make a scatterplot of the Income and Loan Sanction Amount.
# Make the dots small
plt.scatter(dataframe_no_income_outliers['Income (USD)'], dataframe_no_income_outliers['Loan Sanction Amount (USD)'], color='blue', s=0.2)
plt.xlabel("Income (USD)")
plt.ylabel("Loan Sanction Amount (USD)")

# Plot the best fit line
x = np.array([0, 20000])
y = slope * x + intercept
plt.plot(x, y, color='red')

plt.tight_layout()
plt.show()
#
# # Make a best fit line for the Property Price and Loan Sanction Amount and plot it along with the scatterplot
# # Get the slope and intercept of the best fit line
# slope, intercept = np.polyfit(dataframe['Property Price'], dataframe['Loan Sanction Amount (USD)'], 1)

# Make a scatterplot of the Loan Amount Request (USD) and Loan Sanction Amount.
# Make the dots small
plt.scatter(dataframe['Loan Amount Request (USD)'], dataframe['Loan Sanction Amount (USD)'], color='red', s=0.2)
plt.xlabel("Loan Amount Request (USD)")
plt.ylabel("Loan Sanction Amount (USD)")
plt.tight_layout()

# Add an x = y line to compare them with the maximum possible loan sanction amount
x = np.array([0, 500000])
y = x
plt.plot(x, y, color='blue')
plt.show()

no_defaults = dataframe[dataframe['No. of Defaults'] == 0]
defaults = dataframe[dataframe['No. of Defaults'] > 0]
# Make a matrix comparing the rejection rates for defaults and no defaults
rejection_rates = np.array([[len(no_defaults[no_defaults['Loan Sanction Amount (USD)'] == 0]), len(no_defaults[no_defaults['Loan Sanction Amount (USD)'] > 0])],
                            [len(defaults[defaults['Loan Sanction Amount (USD)'] == 0]), len(defaults[defaults['Loan Sanction Amount (USD)'] > 0])]])

# Make a plot of the rejection rates with descriptive labels
# Show the values in each square of the matrix
plt.matshow(rejection_rates)
plt.xlabel("Loan Sanctioned")
plt.ylabel("Defaulted")

# Add the numbers to the plot
for i in range(2):
    for j in range(2):
        # Print white text for dark backgrounds and black text for light backgrounds
        plt.text(x=j, y=i, s=rejection_rates[i, j], va='center', ha='center', size='xx-large', color='white' if rejection_rates[i, j] < 6000 else 'black')
plt.show()

# Make a plot of the property price and loan sanction amounts
# Add a best fit line
# Make the dots small
slope, intercept = np.polyfit(dataframe['Property Price'], dataframe['Loan Sanction Amount (USD)'], 1)

plt.scatter(dataframe['Property Price'], dataframe['Loan Sanction Amount (USD)'], color='red', s=0.2)
plt.xlabel("Property Price")
plt.ylabel("Loan Sanction Amount (USD)")

# Plot the best fit line
x = np.array([0, 500000])
y = slope * x + intercept
plt.plot(x, y, color='blue')

plt.tight_layout()
plt.show()

# Make a plot of the current loan expenses and loan sanction amounts
# Add a best fit line
# Make the dots small
# Replace missing and negative values with 0
dataframe['Current Loan Expenses (USD)'] = dataframe['Current Loan Expenses (USD)'].fillna(0)
dataframe['Current Loan Expenses (USD)'] = dataframe['Current Loan Expenses (USD)'].apply(lambda x: 0 if x < 0 else x)

slope, intercept = np.polyfit(dataframe['Current Loan Expenses (USD)'], dataframe['Loan Sanction Amount (USD)'], 1)

plt.scatter(dataframe['Current Loan Expenses (USD)'], dataframe['Loan Sanction Amount (USD)'], color='red', s=0.2)
plt.xlabel("Current Loan Expenses (USD)")
plt.ylabel("Loan Sanction Amount (USD)")

# Plot the best fit line
x = np.array([0, 3500])
y = slope * x + intercept
plt.plot(x, y, color='blue')

plt.tight_layout()
plt.show()

base_loan_rejection_rate = len(dataframe[dataframe['Loan Sanction Amount (USD)'] <= 0]) / len(dataframe)
print("Base Loan Rejection Rate: ", base_loan_rejection_rate)

# Calculate the loan rejection rate for those who don't possess property
no_property = dataframe[dataframe['Property Price'] <= 0]
no_property_rejection_rate = len(no_property[no_property['Loan Sanction Amount (USD)'] <= 0]) / len(no_property)
print("No Property Rejection Rate: ", no_property_rejection_rate)

property = dataframe[dataframe['Property Price'] > 0]
property_rejection_rate = len(property[property['Loan Sanction Amount (USD)'] <= 0]) / len(property)
print("Property Rejection Rate: ", property_rejection_rate)

credit_score_580_to_650 = dataframe[(dataframe['Credit Score'] >= 580) & (dataframe['Credit Score'] < 650)]
credit_score_580_to_650_rejection_rate = len(credit_score_580_to_650[credit_score_580_to_650['Loan Sanction Amount (USD)'] <= 0]) / len(credit_score_580_to_650)
print("Credit Score 580 to 650 Rejection Rate: ", credit_score_580_to_650_rejection_rate)

credit_score_650_to_700 = dataframe[(dataframe['Credit Score'] >= 650) & (dataframe['Credit Score'] < 700)]
credit_score_650_to_700_rejection_rate = len(credit_score_650_to_700[credit_score_650_to_700['Loan Sanction Amount (USD)'] <= 0]) / len(credit_score_650_to_700)
print("Credit Score 650 to 700 Rejection Rate: ", credit_score_650_to_700_rejection_rate)

credit_score_700_to_750 = dataframe[(dataframe['Credit Score'] >= 700) & (dataframe['Credit Score'] < 750)]
credit_score_700_to_750_rejection_rate = len(credit_score_700_to_750[credit_score_700_to_750['Loan Sanction Amount (USD)'] <= 0]) / len(credit_score_700_to_750)
print("Credit Score 700 to 750 Rejection Rate: ", credit_score_700_to_750_rejection_rate)

credit_score_750_to_800 = dataframe[(dataframe['Credit Score'] >= 750) & (dataframe['Credit Score'] < 800)]
credit_score_750_to_800_rejection_rate = len(credit_score_750_to_800[credit_score_750_to_800['Loan Sanction Amount (USD)'] <= 0]) / len(credit_score_750_to_800)
print("Credit Score 750 to 800 Rejection Rate: ", credit_score_750_to_800_rejection_rate)

credit_score_800_to_850 = dataframe[(dataframe['Credit Score'] >= 800) & (dataframe['Credit Score'] <= 850)]
credit_score_800_to_850_rejection_rate = len(credit_score_800_to_850[credit_score_800_to_850['Loan Sanction Amount (USD)'] <= 0]) / len(credit_score_800_to_850)
print("Credit Score 800 to 850 Rejection Rate: ", credit_score_800_to_850_rejection_rate)

credit_score_850_to_900 = dataframe[(dataframe['Credit Score'] >= 850) & (dataframe['Credit Score'] <= 900)]
credit_score_850_to_900_rejection_rate = len(credit_score_850_to_900[credit_score_850_to_900['Loan Sanction Amount (USD)'] <= 0]) / len(credit_score_850_to_900)
print("Credit Score 850 to 900 Rejection Rate: ", credit_score_850_to_900_rejection_rate)

# Calculate loan rejection rate for those between 60 and 65
between_60_and_65 = dataframe[(dataframe['Age'] >= 60) & (dataframe['Age'] < 65)]
between_60_and_65_rejection_rate = len(between_60_and_65[between_60_and_65['Loan Sanction Amount (USD)'] <= 0]) / len(between_60_and_65)
print("Between 60 and 65 Rejection Rate: ", between_60_and_65_rejection_rate)

# Calculate loan rejection rate for those over 65
over_65 = dataframe[dataframe['Age'] >= 65]
over_65_rejection_rate = len(over_65[over_65['Loan Sanction Amount (USD)'] <= 0]) / len(over_65)
print("Over 65 Rejection Rate: ", over_65_rejection_rate)

# Calculate loan rejection rate for those under 20
under_20 = dataframe[dataframe['Age'] <= 20]
under_20_rejection_rate = len(under_20[under_20['Loan Sanction Amount (USD)'] <= 0]) / len(under_20)
print("Under 20 Rejection Rate: ", under_20_rejection_rate)

# Plot graph of requested loan amount vs age
plt.scatter(dataframe['Age'], dataframe['Loan Amount Request (USD)'], color='red', s=0.2)
plt.xlabel("Age")
plt.ylabel("Loan Amount Request (USD)")
plt.tight_layout()
plt.show()