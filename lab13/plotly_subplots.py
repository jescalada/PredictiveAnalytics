import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH     = "\\..\\datasets\\"
CSV_FILE = 'employee_turnover.csv'
df       = pd.read_csv(ROOT_PATH + PATH + CSV_FILE)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# create 1 by 2 figure of 2 subplots
fig = make_subplots(rows=1, cols=2,
                    # indicate the position of the first plot
                    start_cell="bottom-left",
                    subplot_titles=["Age", "Gender"])

# create histograms and add them to the figure
fig.add_trace(go.Histogram(
    x=df['age'],
    name='Age',
    marker=dict(color='crimson'),
    xbins=dict(start=0, end=100, size=0.1)),  # define bins
    row=1, col=1)

fig.add_trace(go.Histogram(
    x=df['gender'],
    name='Gender',
    marker=dict(color='royalblue'),
    xbins=dict(start=0, end=2, size=0.1)),
    row=1, col=2)

# Update x-axis properties
fig.update_xaxes(title_text="Age", row=1, col=1)
fig.update_xaxes(title_text="Gender", row=1, col=2)

# Update y-axis properties
fig.update_yaxes(title_text="Number of Employees", row=1, col=1)
fig.update_yaxes(title_text="Number of Employees", row=1, col=2)

# show the histograms
fig.show()
