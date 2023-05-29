import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import glob

fnames = glob.glob('./data/*.csv')

df_list = []
for f in fnames:
    df_list.append(pd.read_csv(f, index_col=False))

df = pd.concat(df_list)


'---'
import plotly.express as px

fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", hover_name="City",
                        hover_data=["CO2 emission (Mton/yr)"],
                        color="Sector", zoom=3, height=500)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig)

'---'

df
# fig = go.Figure(go.Scattermapbox(
#     fill = "toself",
#     lon = [-74, -70, -70, -74], lat = [47, 47, 45, 45],
#     marker = { 'size': 10, 'color': "orange" }))
#
# fig.update_layout(
#     mapbox = {
#         'style': "open-street-map", #stamen-terrain",
#         'center': {'lon': -73, 'lat': 46 },
#         'zoom': 5},
#     showlegend = False)
#
# st.plotly_chart(fig)


'---'

