from pandas_datareader import data as pdr
import yfinance as yfin
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Do not show warning.
pd.options.mode.chained_assignment = None  # default='warn'

##################################################################
# CONFIGURATION SECTION
NUM_DAYS = 1200
NUM_TIME_STEPS = 2
TEST_DAYS = 30


##################################################################
# Creates time shifted columns for as many time steps needed.
def back_shift_columns(df, original_col_name, num_time_steps):
    df_new = df[[original_col_name]].pct_change()

    for i in range(1, num_time_steps + 1):
        new_col_name = original_col_name[0] + 't-' + str(i)
        df_new[new_col_name] = df_new[original_col_name].shift(periods=i)
    return df_new


def prepare_stock_df(stock_symbol, columns):
    df = get_stock(stock_symbol, NUM_DAYS)

    # Create data frame with back shift columns for all features of interest.
    merged_df = pd.DataFrame()
    for i in range(0, len(columns)):
        back_shifted_df = back_shift_columns(df, columns[i], NUM_TIME_STEPS)
        if i == 0:
            merged_df = back_shifted_df
        else:
            merged_df = merged_df.merge(back_shifted_df, left_index=True,
                                        right_index=True)

    new_columns = list(merged_df.keys())

    # Append stock symbol to column names.
    for i in range(0, len(new_columns)):
        merged_df.rename(columns={new_columns[i]: stock_symbol + \
                                                  "_" + new_columns[i]}, inplace=True)
    return merged_df


columns = ['Open', 'Close']
msft_df = prepare_stock_df('MSFT', columns)
aapl_df = prepare_stock_df('AAPL', columns)
merged_df = msft_df.merge(aapl_df, left_index=True, right_index=True)
merged_df = merged_df.dropna()
print(merged_df)

corr = merged_df.corr()
plt.figure(figsize=(4, 4))
ax = sns.heatmap(corr[['MSFT_Open']].sort_values(by='MSFT_Open', ascending=False),
                 linewidth=0.5, vmin=-1, annot=True,
                 vmax=1, cmap="YlGnBu")
plt.tight_layout()
plt.show()
