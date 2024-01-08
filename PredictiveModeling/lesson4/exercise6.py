import pandas as pd
import matplotlib.pyplot as plt
import os.path

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "babysamp-98.txt"
dataset = pd.read_csv(PATH + CSV_DATA, sep='\t')

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# ------------------------------------------------------------------
# Show statistics, boxplot, extreme values and returns DataFrame
# row indexes where outliers exist outside an upper and lower percentile.
# ------------------------------------------------------------------
def view_and_get_outliers_by_percentile(df, col_name, lower_p, upper_p, plot):
    # Show basic statistics.
    df_sub = df[[col_name]]
    print("*** Statistics for " + col_name)
    print(df_sub.describe())

    # Show boxplot.
    df_sub.boxplot(column=[col_name])
    plot.title(col_name)
    plot.show()

    # Get upper and lower percentiles and filter with them.
    up = df[col_name].quantile(upper_p)
    lp = df[col_name].quantile(lower_p)
    outlier_df = df[(df[col_name] < lp) | (df[col_name] > up)]

    # Show filtered and sorted DataFrame with outliers.
    df_sorted = outlier_df.sort_values([col_name], ascending=[True])
    print("\nDataFrame rows containing outliers for " + col_name + ":")
    print(df_sorted)

    return lp, up  # return lower and upper percentiles


LOWER_PERCENTILE = 0.02
UPPER_PERCENTILE = 0.98
lp, up = view_and_get_outliers_by_percentile(dataset, 'gestation',
                                             LOWER_PERCENTILE, UPPER_PERCENTILE, plt)
