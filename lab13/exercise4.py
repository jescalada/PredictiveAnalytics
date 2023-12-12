import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = "\\..\\datasets\\"
CSV_FILE = 'babysamp-98.txt'

df = pd.read_csv(ROOT_PATH + PATH + CSV_FILE, sep='\t')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# create 1 by 2 figure of 2 subplots
fig = make_subplots(rows=1, cols=3,
                    # indicate the position of the first plot
                    start_cell="bottom-left",
                    subplot_titles=["Mom Age", "Mom Education", "Mom Married"])

# create histograms and add them to the figure
fig.add_trace(go.Histogram(
    x=df['MomAge'],
    name='Mom Age',
    marker=dict(color='crimson'),
    xbins=dict(start=0, end=100, size=0.1)),  # define bins
    row=1, col=1)

fig.add_trace(go.Histogram(
    x=df['MomEduc'],
    name='Mom Education',
    marker=dict(color='royalblue'),
    xbins=dict(start=0, end=20, size=1)),
    row=1, col=2)

fig.add_trace(go.Histogram(
    x=df['MomMarital'],
    name='Mom Married',
    marker=dict(color='yellow'),
    xbins=dict(start=0, end=2, size=0.1)),
    row=1, col=3)

# Update x-axis properties
fig.update_xaxes(title_text="Mom Age", row=1, col=1)
fig.update_xaxes(title_text="Mom Education", row=1, col=2)
fig.update_xaxes(title_text="Mom Married", row=1, col=2)

# Update y-axis properties
fig.update_yaxes(title_text="Number of Babies", row=1, col=1)
fig.update_yaxes(title_text="Number of Babies", row=1, col=2)
fig.update_yaxes(title_text="Number of Babies", row=1, col=2)

# show the histograms
fig.show()
