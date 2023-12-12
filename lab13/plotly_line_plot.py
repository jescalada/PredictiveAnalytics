import plotly.express as px
import pandas as pd

years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
colorado = [5029196, 5029316, 5048281, 5121771, 5193721, 5270482, 5351218, 5452107,
            5540921, 5615902, 5695564]
conneticut = [3574097, 3574147, 3579125, 3588023, 3594395, 3594915, 3594783, 3587509,
              3578674, 3573880, 3572665]

colorado_df = pd.DataFrame(data={
    "year": years, "population": colorado
})
colorado_df['state'] = "Colorado"

conn_df = pd.DataFrame(data={"year": years, "population": conneticut})
conn_df['state'] = "Conneticut"
df = colorado_df._append(conn_df)
print(df)

fig = px.line(df, x="year", y="population", color="state")

fig.update_traces(mode="markers+lines")

fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)

fig.show()
