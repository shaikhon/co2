import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import glob


fnames = glob.glob('./data/*.csv')
fnames

# df = pd.read_csv('./data/')



'---'

fig = go.Figure(go.Scattermapbox(
    fill = "toself",
    lon = [-74, -70, -70, -74], lat = [47, 47, 45, 45],
    marker = { 'size': 10, 'color': "orange" }))

fig.update_layout(
    mapbox = {
        'style': "open-street-map", #stamen-terrain",
        'center': {'lon': -73, 'lat': 46 },
        'zoom': 5},
    showlegend = False)

st.plotly_chart(fig)


'---'

