import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "USA_Housing.csv"
dataset = pd.read_csv(PATH + CSV_DATA)

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(dataset.head())


# ------------------------------------------------------------------
# Show statistics, boxplot, extreme values and returns DataFrame
# row indexes where outliers exist.
# ------------------------------------------------------------------
def view_and_get_outliers(df, col_name, threshold, plot):
    # Show basic statistics.
    df_sub = df[[col_name]]
    print("*** Statistics for " + col_name)
    print(df_sub.describe())

    # Show boxplot.
    df_sub.boxplot(column=[col_name])
    plot.title(col_name)
    plot.show()

    # Note this is absolute 'abs' so it gets both high and low values.
    z = np.abs(stats.zscore(df_sub))
    row_column_array = np.where(z > threshold)
    row_indices = row_column_array[0]

    # Show outlier rows.
    print("\nOutlier row indexes for " + col_name + ":")
    print(row_indices)
    print("")

    # Show filtered and sorted DataFrame with outliers.
    df_sub = df.iloc[row_indices]
    df_sorted = df_sub.sort_values([col_name], ascending=[True])
    print("\nDataFrame rows containing outliers for " + col_name + ":")
    print(df_sorted)
    return row_indices


THRESHOLD_Z = 3
price_outlier_rows = view_and_get_outliers(dataset, 'Avg. Area Income', THRESHOLD_Z, plt)
