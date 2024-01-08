import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = f"{ROOT_PATH}\\..\\datasets\\"

FILE = "carPrice.csv"
df = pd.read_csv(DATASET_DIR + FILE)

# Enable the display of all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# ---------------------------------------------
# Generate quick views of data.
def view_quick_stats():
    print("\n*** Show contents of the file.")
    print(df.head())

    print("\n*** Show the description for all columns.")
    print(df.info())

    print("\n*** Describe numeric values.")
    print(df.describe())

    print("\n*** Showing frequencies.")

    # Show frequencies.
    print(df['model'].value_counts())
    print("")
    print(df['transmission'].value_counts())
    print("")
    print(df['fuel type'].value_counts())
    print("")
    print(df['engine size'].value_counts())
    print("")
    print(df['fuel type2'].value_counts())
    print("")
    print(df['year'].value_counts())
    print("")


# ---------------------------------------------
# Fix the price column.
for i in range(0, len(df)):
    priceStr = str(df.iloc[i]['price'])
    priceStr = priceStr.replace("£", "")
    priceStr = priceStr.replace("-", "")
    priceStr = priceStr.replace(",", "")
    df.at[i, 'price'] = priceStr

# Convert column to number.
df['price'] = pd.to_numeric(df['price'])

# ---------------------------------------------
# Fix the price column.
averageYear = df['year'].mean()
for i in range(0, len(df)):
    year = df.iloc[i]['year']

    if np.isnan(year):
        df.at[i, 'year'] = averageYear

# ---------------------------------------------
# Fix the model column.
for i in range(0, len(df)):
    df.at[i, 'model'] = model = df.iloc[i]['model'].strip()

# ---------------------------------------------
# Fix the engine size2 column.
for i in range(0, len(df)):
    try:
        engineSize2 = df.loc[i]['engine size2']
        if pd.isna(engineSize2):
            df.at[i, 'engine size2'] = "0"

    except Exception as e:
        error = str(e)
        print(error)

df['engine size2'] = pd.to_numeric(df['engine size2'])
df['mileage2'].value_counts()
view_quick_stats()

# ---------------------------------------------
# Fix the mileage column.
for i in range(0, len(df)):
    mileageStr = str(df.iloc[i]['mileage'])
    mileageStr = mileageStr.replace(",", "")
    df.at[i, 'mileage'] = mileageStr
    try:
        if not mileageStr.isnumeric():
            df.at[i, 'mileage'] = "0"
    except Exception as e:
        error = str(e)
        print(error)

df['mileage'] = pd.to_numeric(df['mileage'])

view_quick_stats()

# Create bins for years
df['year_bin'] = pd.cut(df['year'], bins=[1991, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
# Isolate year bins
temp_df = df[['year_bin', 'transmission', 'fuel type2']]

# Get dummies
dummy_df = pd.get_dummies(temp_df, columns=['year_bin', 'transmission', 'fuel type2'], dtype=int)
df = pd.concat(([df, dummy_df]), axis=1)  # Join dummy df with original

print(df.head(10))

# Get a dataframe without the non-numeric columns.
df2 = df.select_dtypes(include=np.number)

# Compute the correlation matrix
corr = df2.corr()

# plot the heatmap
sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns)
plt.show()

X = df[
    ['transmission_Automatic', 'transmission_Manual', 'transmission_Other',
     'transmission_Semi-Auto', 'fuel type2_Petrol',
     'year_bin_(1991, 2013]', 'year_bin_(2013, 2014]', 'year_bin_(2014, 2015]', 'year_bin_(2015, 2016]',
     'year_bin_(2016, 2017]', 'year_bin_(2017, 2018]', 'year_bin_(2018, 2019]', 'mileage']
].values

from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm

# Adding an intercept *** This is requried ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X = sm.add_constant(X)

y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)  # make the predictions by the model

print(model.summary())

# Get the coefficients.
print('Coefficients: \n', model.params)

print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))


def estimate_car_price(actual_price: int, year: int, transmission_type: str, fuel_type: str, mileage: int) -> float:
    transmission_automatic, transmission_manual, transmission_other, transmission_semi_auto = 0, 0, 0, 0
    if transmission_type.strip().lower() == "automatic":
        transmission_automatic = 1
    elif transmission_type.strip().lower() == "manual":
        transmission_manual = 1
    elif transmission_type.strip().lower() == "other":
        transmission_other = 1
    elif transmission_type.strip().lower() == "semi-auto":
        transmission_semi_auto = 1

    fuel_type_petrol = 0
    if fuel_type.strip().lower() == "petrol":
        fuel_type_petrol = 1

    year_bin_1991_2013, year_bin_2013_2014, year_bin_2014_2015, year_bin_2015_2016, year_bin_2016_2017, year_bin_2017_2018, year_bin_2018_2019 = 0, 0, 0, 0, 0, 0, 0
    if 1991 <= year <= 2013:
        year_bin_1991_2013 = 1
    elif 2013 < year <= 2014:
        year_bin_2013_2014 = 1
    elif 2014 < year <= 2015:
        year_bin_2014_2015 = 1
    elif 2015 < year <= 2016:
        year_bin_2015_2016 = 1
    elif 2016 < year <= 2017:
        year_bin_2016_2017 = 1
    elif 2017 < year <= 2018:
        year_bin_2017_2018 = 1
    elif 2018 < year <= 2019:
        year_bin_2018_2019 = 1

    return 2.71271664e+04 \
           + 8.59928946e+03 * transmission_automatic \
           + 4.64903692e+03 * transmission_manual \
           + 5.75738956e+03 * transmission_other \
           + 8.12145050e+03 * transmission_semi_auto \
           + 2.20917711e+03 * fuel_type_petrol \
           - 2.08895560e+04 * year_bin_1991_2013 \
           - 1.67855074e+04 * year_bin_2013_2014 \
           - 1.57255962e+04 * year_bin_2014_2015 \
           - 1.37321635e+04 * year_bin_2015_2016 \
           - 1.19527997e+04 * year_bin_2016_2017 \
           - 1.08031557e+04 * year_bin_2017_2018 \
           - 6.48752867e+03 * year_bin_2018_2019 \
           - 9.85112460e-02 * mileage


# C Class,2013," £9,995",Automatic,"44,900",29,£160,46.3,Petrol,1.6,/ad/25128085
# C Class,2012," £6,995",Automatic,"88,200",34,£125,58.9,Diesel,2.1,/ad/23763610
# C Class,2012," £7,495",Automatic,"115,000",37,£145,54.3,Diesel,2.1,/ad/24626603
print(f"Car 1: Actual: 9995, Estimated: {estimate_car_price(9995, 2013, 'Automatic', 'Petrol', 44900)}")
print(f"Car 2: Actual: 6995, Estimated: {estimate_car_price(6995, 2012, 'Automatic', 'Diesel', 88200)}")
print(f"Car 3: Actual: 7495, Estimated: {estimate_car_price(7495, 2012, 'Automatic', 'Diesel', 115000)}")
