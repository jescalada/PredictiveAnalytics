import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_DIR}\\..\\datasets\\"
FILE = 'shampoo.csv'
df   = pd.read_csv(PATH + FILE, index_col=0)
df.info()

# Plot data before differencing.
df.plot()
plt.xticks(rotation=45)
plt.show()

# Perform differencing.
df_differenced = df.diff()

# Plot data after differencing.
plt.plot(df_differenced)
plt.xticks(rotation=75)
plt.show()
