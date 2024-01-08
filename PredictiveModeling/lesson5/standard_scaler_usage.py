import pandas as pd
from sklearn.preprocessing import StandardScaler
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
def show_automated_scaler_results(X):
    sc_x = StandardScaler()
    x_scale = sc_x.fit_transform(X)
    salary = X.iloc[0][1]
    scaled_salary = x_scale[0][1]  # Get first scaled salary.
    print("The first unscaled salary in the list is: " + str(salary))
    print("$19,000 scaled using StandardScaler() is: " + str(scaled_salary))


def get_sd_with_zero_degrees_freedom(X):
    mean = X['EstimatedSalary'].mean()

    # StandardScaler calculates the standard deviation with zero degrees of freedom.
    s1 = df['EstimatedSalary'].std(ddof=0)
    print("sd with 0 degrees of freedom automated: " + str(s1))

    # This is the same calculation manually. (**2 squares the result)
    s2 = np.sqrt(np.sum(((X['EstimatedSalary'] - mean) ** 2)) / (len(X)))
    print("sd with 0 degrees of freedom manually:  " + str(s2))

    return s1


print("*** Showing automated results: ")
show_automated_scaler_results(X)

print("\n*** Showing manually calculated results: ")
sd = get_sd_with_zero_degrees_freedom(X)
mean = df['EstimatedSalary'].mean()
scaled = (19000 - mean) / sd

print("$19,000 scaled manually is: " + str(scaled))

scaled2 = (20000 - mean) / sd
print("$20,000 scaled manually is: " + str(scaled2))
