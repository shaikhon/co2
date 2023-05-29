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
order={'Sector':['Electricity', 'Desalination','Petrochemicals','Refinery','Cement','Steel']}
colors=['red','blue','yellow','green','cyan','purple']

fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", hover_name="City",
                        hover_data=["CO2 emission (Mton/yr)"],
                        size="CO2 emission (Mton/yr)",
                        size_max=30,
                        category_orders=order,
                        color="Sector",
                        color_discrete_sequence=colors, zoom=4.2, width=800, height=600)
fig.update_layout(mapbox_style="carto-positron")  #carto-positron
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

