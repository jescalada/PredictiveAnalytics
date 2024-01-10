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


import matplotlib.pyplot as plt
def showStocks(df, stock, title):
    plt.plot(df.index, df['Close'], label=stock)
    plt.xticks(rotation=70)

NUM_DAYS = 20
df = get_stock('AMZN', NUM_DAYS)
df['Close'] = df['Close'].pct_change()
showStocks(df,'AMZN',"AMZN Close Prices")

df = get_stock('AAPL', NUM_DAYS)
df['Close'] = df['Close'].pct_change()
showStocks(df, 'AAPL', "AAPL Close Prices")

df = get_stock('MSFT', NUM_DAYS)
df['Close'] = df['Close'].pct_change()
showStocks(df, 'MSFT', "MSFT Close Prices")

# Make graphs appear.
plt.legend()
plt.show()
