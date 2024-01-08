import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = "\\..\\datasets\\"
CSV_FILE = 'employee_turnover.csv'
df = pd.read_csv(ROOT_PATH + PATH + CSV_FILE)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def highlight_greaterthan_5(column):
    if column.anxiety < 3.0:
        return ['background-color: green']
    elif column.anxiety < 5.0:
        return ['background-color: yellow']
    else:
        return ['background-color: red']


df.head(5).style \
    .background_gradient(cmap='Oranges', subset=['experience']) \
    .apply(highlight_greaterthan_5, axis=1, subset=["anxiety"])

print(df)
