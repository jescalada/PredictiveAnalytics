from pandas_datareader import data as pdr
import yfinance as yfin  # Work around until pandas_datareader is fixed.
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def get_stock(stk, total_days):
    num_days = int(total_days)
    # Only gets up until day before during
    # trading hours
    dt = datetime.date.today()
    # For some reason, must add 1 day to get current stock prices
    # during trade hours. (Prices are about 15 min behind actual prices.)
    dt_now = dt + datetime.timedelta(days=1)
    dt_now_str = dt_now.strftime("%Y-%m-%d")
    dt_past = dt + datetime.timedelta(days=-num_days)
    dt_past_str = dt_past.strftime("%Y-%m-%d")
    yfin.pdr_override()
    df = pdr.get_data_yahoo(stk, start=dt_past_str, end=dt_now_str)
    return df


def show_stock(df, title):
    plt.plot(df.index, df['Close'])
    plt.title(title)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()

# Get Southwestern stock for last 60 days
NUM_DAYS = 1200
df = get_stock('LUV', NUM_DAYS)
print("South West Airlines")
print(df)

# Create weekly summary of closing price standard deviations
from pandas.tseries.frequencies import to_offset

series = df['Close'].resample('M').std()
series.index = series.index + to_offset("-1D")
summaryDf = series.to_frame()

# Convert datetime index to date and then graph it.
summaryDf.index = summaryDf.index.date
print("Summary of monthly standard deviations")
print(summaryDf)
show_stock(summaryDf, "Monthly S.D. Southwest Airlines")
