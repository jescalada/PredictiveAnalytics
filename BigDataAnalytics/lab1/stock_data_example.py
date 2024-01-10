from pandas_datareader import data as pdr
import yfinance as yfin  # Work around until
# pandas_datareader is fixed.
import pandas as pd
import datetime

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def get_stock(stk, ttlDays):
    numDays = int(ttlDays)
    # Only gets up until day before during
    # trading hours
    dt = datetime.date.today()
    # For some reason, must add 1 day to get current stock prices
    # during trade hours. (Prices are about 15 min behind actual prices.)
    dtNow = dt + datetime.timedelta(days=1)
    dtNowStr = dtNow.strftime("%Y-%m-%d")
    dtPast = dt + datetime.timedelta(days=-numDays)
    dtPastStr = dtPast.strftime("%Y-%m-%d")
    yfin.pdr_override()
    df = pdr.get_data_yahoo(stk, start=dtPastStr, end=dtNowStr)
    return df


NUM_DAYS = 1000
# Search Yahoo for the correct symbols.
df = get_stock('MSFT', NUM_DAYS)
print("Microsoft stock")
print(df)

# Canadian stocks have the suffix
# .TO, .V or .CN for Canadian markets.
NUM_DAYS = 1000
df = get_stock('TD.TO', NUM_DAYS)
print("TD Bank stock")
print(df)

# Show the last 10 rows
print(df.tail(10))

import matplotlib.pyplot as plt

def show_stock(df, title):
    plt.plot(df.index, df['Close'])
    plt.title(title)
    plt.xticks(rotation=70)
    plt.show()

show_stock(df, "TD Bank (TD.TO) Close Prices")
