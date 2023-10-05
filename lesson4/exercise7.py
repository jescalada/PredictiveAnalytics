import pandas as pd
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV_DATA = "wnba.csv"
df = pd.read_csv(PATH + CSV_DATA,
                      skiprows=1,  # Don't include header row as part of data.
                      encoding="ISO-8859-1", sep=',', names=('PLAYER', 'GP', 'PTS'))
print(df.head(30))

df_pts_adjusted = df['PTS'].clip(upper=860)
df_games_adjusted = df['GP'].clip(upper=36)
df['PTS_ADJUSTED'] = df_pts_adjusted
df['GP_ADJUSTED'] = df_games_adjusted

# View the adjusted dataframe
print(df.head(30))

# Eliminate the outliers
df_filtered = df[(df['PTS'] <= 860) & (df['GP'] <= 36)]

print(df_filtered)
