from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas_datareader import data as pdr
import yfinance as yfin
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings("ignore")

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def get_stock(stk, ttl_days):
    num_days = int(ttl_days)
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


NUM_DAYS = 70
df = get_stock('MSFT', NUM_DAYS)
print(df)

# Plot ACF for stock.
plot_acf(df['Open'])
plt.show()

plot_pacf(df['Open'])
plt.show()
