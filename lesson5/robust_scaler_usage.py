import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "computerPurchase.csv"
df = pd.read_csv(PATH + CSV_DATA)

# Separate into x and y values.
X = df[["Age", "EstimatedSalary"]]
y = df['Purchased']


# Split data.
def show_automated_scaler_results(X, index: int = 1):
    sc_x = RobustScaler()
    x_scale = sc_x.fit_transform(X)
    salary = X.iloc[index][1]
    scaled_salary = x_scale[index][1]
    print(f"{salary} scaled using StandardScaler() is: {scaled_salary}")

print("*** Showing automated results: ")
show_automated_scaler_results(X)

# Find the index of the lowest salary
min_salary = df['EstimatedSalary'].min()
min_salary_index = df['EstimatedSalary'].idxmin()

show_automated_scaler_results(X, min_salary_index)

print(df.describe())
