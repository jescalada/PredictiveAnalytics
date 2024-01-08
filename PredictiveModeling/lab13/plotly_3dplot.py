import plotly.express as px

df = px.data.iris()
fig_3d = px.scatter_3d(
    df, x='sepal_width', y='sepal_length', z='petal_width',
    color='species')

fig_3d.show()
