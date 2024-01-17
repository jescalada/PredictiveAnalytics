from pandas_datareader import data as pdr
import yfinance as yfin
import datetime
import matplotlib.pyplot as plt


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


df = get_stock('MSFT', 1100)
print(df)

rolling_mean2 = df['Close'].rolling(window=50).mean()
exponential_moving_average_low_alpha = df['Close'].ewm(alpha=0.05).mean()
exponential_moving_average_high_alpha = df['Close'].ewm(alpha=0.4).mean()

# plt.figure(figsize=(10,30))
df['Close'].plot(label='MSFT Close ', color='gray', alpha=0.3)
rolling_mean2.plot(label='MSFT 50 Day SMA', style='--', color='magenta')
exponential_moving_average_low_alpha.plot(label='MSFT EMA (low alpha)', style='--', color='blue')
exponential_moving_average_high_alpha.plot(label='MSFT EMA (high alpha)', style='--', color='red')

plt.legend()
plt.show()
