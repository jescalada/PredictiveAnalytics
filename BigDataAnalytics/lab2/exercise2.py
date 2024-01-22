from pandas_datareader import data as pdr
import yfinance as yfin  # Work around until
# pandas_datareader is fixed.
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
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


def get_new_balance(start_balance, start_price, end_price):
    qty = int(start_balance / start_price)
    cash_left_over = start_balance - qty * start_price
    end_value = qty * end_price
    balance = cash_left_over + end_value
    return balance


def show_buy_and_hold_earnings(df, balance):
    start_close_price = df.iloc[0]['Close']
    end_close_price = df.iloc[len(df) - 1]['Close']
    new_balance = get_new_balance(balance, start_close_price, end_close_price)
    print("Buy and hold closing balance: $" + str(round(new_balance, 2)))


def show_strategy_earnings(df, balance, lt, st):
    buy_price = 0
    buy_date = None
    sell_date = None
    bought = False

    buy_sell_dates = []
    prices = []

    df_strategy = pd.DataFrame(columns=['buyDt', 'buy$', 'sellDt',
                                        'sell$', 'balance'])
    dates = list(df.index)
    for i in range(0, len(df)):
        if (df.iloc[i]['Buy'] and not bought):
            buy_price = df.iloc[i]['Close']
            buy_date = dates[i]
            bought = True
            buy_sell_dates.append(buy_date)
            prices.append(buy_price)

        elif (df.iloc[i]['Sell'] and bought):
            sell_price = df.iloc[i]['Close']
            balance = get_new_balance(balance, buy_price, sell_price)
            sell_date = dates[i]
            buy_sell_info = {'buyDt': buy_date, 'buy$': buy_price,
                             'sellDt': sell_date, 'sell$': sell_price,
                             'balance': balance, }
            df_strategy = df_strategy._append(buy_sell_info, ignore_index=True)
            bought = False
            buy_sell_dates.append(sell_date)
            prices.append(sell_price)

    print(df_strategy)
    print("\nMoving average strategy closing balance: $" + str(round(balance, 2)))
    return buy_sell_dates, prices


def show_buy_and_sell_dates(df, start_balance):
    strategy_dates, strategy_prices = show_strategy_earnings(df, start_balance, lt, st)
    plt.plot(df.index, df['Close'], label='Close')
    plt.plot(df.index, df['ema20'], label='ema20', alpha=0.4)
    plt.plot(df.index, df['ema50'], label='ema50', alpha=0.4)
    plt.scatter(strategy_dates, strategy_prices, label='Buy/Sell', color='red')
    plt.xticks(rotation=70)
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_investment_differences(df_stock, lt, st):
    df = df_stock.copy()
    df['ema50'] = df['Close'].ewm(span=lt).mean()
    df['ema20'] = df['Close'].ewm(span=st).mean()

    # Remove nulls.
    df.dropna(inplace=True)
    df.round(3)
    own_positions = np.where(df['ema20'] > df['ema50'], 1, 0)
    df['Position'] = own_positions
    df.round(3)

    df['Buy'] = (df['Position'] == 1) & (df['Position'].shift(1) == 0)
    df['Sell'] = (df['Position'] == 0) & (df['Position'].shift(1) == 1)

    START_BALANCE = 10000

    print("-------------------------------------------------------")
    show_buy_and_hold_earnings(df, START_BALANCE)
    print("-------------------------------------------------------")
    show_buy_and_sell_dates(df, START_BALANCE)


longterms = [75]
shortterms = [50]
df_stock = get_stock('XOM', 1100)

for lt in longterms:
    for st in shortterms:
        print("\b******************************************************")
        print("Lt: " + str(lt))
        print("St: " + str(st))
        show_investment_differences(df_stock, lt, st)
