import pandas as pd
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "\\..\\datasets\\"
CSV_FILE = 'employee_turnover.csv'
df = pd.read_csv(ROOT_PATH + DATASET_DIR + CSV_FILE)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# get count of employees who have left, grouped by profession
left = df[df['turnover'] == 1].groupby('gender')['gender'].count() \
    .reset_index(name="Left Company")

# get count of employees who have stayed, grouped by profession
stayed = df[df['turnover'] == 0].groupby('gender')['gender'].count() \
    .reset_index(name="Still With Company")
stayed = stayed.set_index('gender')

left = left.set_index('gender')

result = pd.merge(left, stayed, on='gender', how='outer')

# Sort by index and replace null values.
result = result.sort_index()
result = result.fillna(0)

# Dropped profession 6 to avoid distorting scale for other professions.
result = result[result.index != 6]

import plotly.express as px

# create the stacked bar chart
fig = px.bar(result,
             x=result.index,
             # specify columns to use for the stacked bars
             y=["Still With Company", "Left Company"],
             title="Turnover By Gender",
             )

# adjust layout properties of the figure
fig.update_layout(
    yaxis=dict(title="Number of Employees"),
    xaxis=dict(title="Gender"),
    width=600,
    height=550,
    hoverlabel=dict(
        bgcolor="yellow",
        font_size=16,
        font_family="Rockwell"
    )
)
# show the stacked bar chart
fig.show()
