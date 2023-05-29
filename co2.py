# import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import glob

st.set_page_config(
    page_title="Net0thon App",
    page_icon=":four_leaf_clover:",
    layout="wide",
    initial_sidebar_state="collapsed",  #expanded
    menu_items={'Get help':'mailto:obai.shaikh@gmail.com',
                'Report a Bug':'mailto:obai.shaikh@gmail.com',
                'About':"The Green Guardian's team effort in Net0thon to combat climate change."}
)

fnames = glob.glob('./data/*.csv')

df_list = []
for f in fnames:
    df_list.append(pd.read_csv(f, index_col=False))

df = pd.concat(df_list)
df.drop(inplace=True, columns=['Primary Fuel','Unit Type'], errors='ignore')

'---'

st.markdown("<h1 style='text-align: center; color: white;'>CO2 Emissions</h1>", unsafe_allow_html=True)

order={'Sector':['Electricity', 'Desalination','Petrochemicals','Refinery','Cement','Steel']}
colors=['red','blue','yellow','green','cyan','purple','orange']

fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", hover_name="City",
                        hover_data=["Sector", "CO2 emission (Mton/yr)"],
                        size="CO2 emission (Mton/yr)",
                        size_max=30,
                        category_orders=order,
                        color="Sector",
                        color_discrete_sequence=colors, zoom=4.2, width=800, height=600)
fig.update_layout( title_text = "2022 Saudi Arabia's CO2 Emissions",
                   mapbox_style="carto-positron")  #carto-positron
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

