# import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import glob
from prophet import Prophet
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Net0thon App",
    page_icon=":four_leaf_clover:",
    layout="wide",
    initial_sidebar_state="collapsed",  #expanded
    menu_items={'Get help':'mailto:obai.shaikh@gmail.com',
                'Report a Bug':'mailto:obai.shaikh@gmail.com',
                'About':"The Green Guardian's team effort in Net0thon to combat climate change."}
)

def co2_map(color_by):
    # Read files
    fnames = glob.glob('./data/by_sector/*.csv')

    df_list = []
    for f in fnames:
        df_list.append(pd.read_csv(f, index_col=False))

    df = pd.concat(df_list)
    # df.drop(inplace=True, columns=['Primary Fuel', 'Unit Type'], errors='ignore')

    # st.markdown("<h1 style='text-align: center; color: white;'>CO2 Emissions</h1>", unsafe_allow_html=True)

    order = {'Sector': ['Electricity', 'Desalination', 'Petrochemicals', 'Refinery', 'Cement', 'Steel']}
    colors = ['red', 'blue', 'yellow', 'green', 'cyan', 'purple', 'orange']

    fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", hover_name="City",
                            hover_data=["Sector", "CO2 emission (Mton/yr)"],
                            size="CO2 emission (Mton/yr)",
                            size_max=30,
                            category_orders=order,
                            color=color_by,
                            color_discrete_sequence=colors,
                            zoom=4,
                            # width=450,
                            height=500)
    fig.update_layout(
        # title_text="2020 Saudi Arabia's CO2 Emissions",
        mapbox_style="carto-positron")  # carto-positron
    fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})

    return fig


def annual_prophecy(d, ys, growth='linear', forecast_period=15):
    '''
    prophecy - FORECAST FUTURE STOCK PRICE

    Inputs:
    d                  Price history DataFrame (yfinance)
    forecast_period    Number of minutes of future forecast to predict

    '''
    d.index.names = ['ds']  # rename index to ds
    d = d.tz_localize(None)  # make timezone naive, for prophet
    ds = d.reset_index()  # make index (ds) a column

    for y in ys:
        ds_temp = ds.loc[:, ['ds', y]].rename(columns={y: 'y'})

        # Make the prophet model and fit on the data
        gm_prophet = Prophet(
            growth=growth,
            changepoints=None,
            n_changepoints=len(ds),
            changepoint_range=.92,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=None,
            seasonality_mode='additive',
            seasonality_prior_scale=0.05,  # yhat zigzag
            holidays_prior_scale=0,
            changepoint_prior_scale=.25, #  .25,  # yhat slope, largefr == steeper
            mcmc_samples=0,
            interval_width=.8,
            uncertainty_samples=1000,
            stan_backend=None
        )

        gm_prophet.fit(ds_temp)
        # predict:
        # Make a future dataframe
        gm_forecast = gm_prophet.make_future_dataframe(periods=forecast_period, freq='Y')
        # Make predictions
        gm_forecast = gm_prophet.predict(gm_forecast)
        # gm_forecast
        gm_forecast = gm_forecast.set_index(gm_forecast.ds).loc[:, ['yhat', 'yhat_lower', 'yhat_upper',
                                                                    'trend', 'trend_lower', 'trend_upper']]
        # merge
        d = gm_forecast.merge(d, how='outer', on='ds')

    return d


def prophet_plot(d):
    color = 'lime'  # if current_price >= open else 'rgb(255, 49, 49)'
    x = d.index.to_list()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # plot population Actual
    fig.add_trace(go.Scatter(mode='lines', x=x, y=d['pop'],
                             line=dict(color='magenta', width=2),
                             hovertemplate='<i>Pop.</i>: %{y:.2f}' +
                                           '<br><i>Year</i>: %{x|%Y}<br><extra></extra>',
                             name='Population',
                             showlegend=True),
                  secondary_y=True)

    # plot population Forecast
    fig.add_trace(go.Scatter(mode='lines', x=x, y=d.yhat_x,
                             line=dict(color='magenta', width=2, dash='dash'),
                             hovertemplate='<i>Pop. Forecast</i>: %{y:.2f}' +
                                           '<br><i>Year</i>: %{x|%Y}<br><extra></extra>',
                             name='Population Forecast',
                             showlegend=True),
                  secondary_y=True)

    # plot (CO2 emissions) Actual
    fig.add_trace(go.Scatter(mode='lines', x=x, y=d.co2_mt,
                             line=dict(color=color, width=2),
                             hovertemplate='<i>CO2</i>: %{y:.2f} million ton' +
                                           '<br><i>Year</i>: %{x|%Y}<br><extra></extra>',
                             name='CO2',
                             showlegend=True),
                  secondary_y=False)

    # plot (CO2 emissions) Forecast
    fig.add_trace(go.Scatter(mode='lines', x=x, y=d.yhat_y,
                             line=dict(color=color, width=2, dash='dash'),
                             hovertemplate='<i>CO2</i>: %{y:.2f} million ton' +
                                           '<br><i>Year</i>: %{x|%Y}<br><extra></extra>',
                             name='CO2 Forecast',
                             showlegend=True),
                  secondary_y=False)

    # plot (CO2 abatment)
    fig.add_trace(go.Scatter(mode='lines', x=x, y=d.abate,
                             line=dict(color='blue', width=2),
                             hovertemplate='<i>Pop.</i>: %{y:.2f}' +
                                           '<br><i>Year</i>: %{x|%Y}<br><extra></extra>',
                             name='Abatement',
                             showlegend=True),
                  secondary_y=False)

    # plot (CO2 abatment) forecast
    fig.add_trace(go.Scatter(mode='lines', x=x, y=d.abate2,
                             line=dict(color='blue', width=2, dash='dash'),
                             hovertemplate='<i>Pop.</i>: %{y:.2f}' +
                                           '<br><i>Year</i>: %{x|%Y}<br><extra></extra>',
                             name='Abatement Estimate',
                             showlegend=True),
                  secondary_y=False)

    fig.add_hline(y=278, line_width=3, line_dash="dash", line_color="green", annotation_text="2030 Goal")

    fig.update_layout(
        # title_text="Saudi Arabia's CO2 & Population Forecast",
        title=dict(text="Saudi Arabia's CO2 & Population Forecast", font=dict(size=32)),
        hovermode="closest",
        hoverlabel=dict(align="left", bgcolor="rgba(0,0,0,0)"),
        #         template="plotly_dark",
        margin=dict(t=40, b=0, l=0, r=0),

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        yaxis=dict(showgrid=False, tickfont = dict(size=18),
                   title={"font": dict(size=20), "text": "CO2 (million ton)", "standoff": 10}),
        yaxis2=dict(showgrid=False, range=[0, 50e6], tickfont = dict(size=18),
                    title={"font": dict(size=20), "text": "Population (million)", "standoff": 10}),
        xaxis=dict(showline=False, tickfont = dict(size=18),)
    )
    return fig


def co2_ml(l3_per_yr=100, growth_rate=1.05):
    fnames = glob.glob('./data/*.csv')
    n = 15  # forecast years
    rate = .15  # tons co2/yr per tree
    growth_vector = np.zeros(n)

    for i, item in enumerate(growth_vector):
        if i == 0:
            growth_vector[i] = rate
            continue
        growth_vector[i] = growth_vector[i - 1] * growth_rate

    l3 = np.cumsum(growth_vector * l3_per_yr) / 1e6  # million tons per year

    # kt of co2
    df_list = []
    for f in fnames:
        df = pd.read_csv(f).T.dropna(subset=[0])
        df_list.append(df)

    df = pd.concat(df_list, axis=1)
    df.rename(columns={0: 'co2_kt', 1: 'pop',
                       2: 'utmn_eor', 3: 'sabic', 4: 'mangrove'},
              inplace=True)
    df['abate'] = df.utmn_eor + df.sabic + df.mangrove + 10
    df['co2_mt'] = df.loc[:, 'co2_kt'] / 1000
    df.index = pd.to_datetime(df.index)

    cols = ['co2_mt', 'pop']

    df = annual_prophecy(df, cols, forecast_period=n)
    df['abate2'] = df.abate.pad()
    # df.abate2.iloc[-n:] += l3
    # df.abate2.iloc[-n:] *= growth_vector / rate

    impact = l3*(100*growth_vector+growth_rate)
    df.abate2.iloc[-n:] += impact

    total_co2 = sum(impact)


    to_target = (df.abate2.iloc[-1]/278)*100

    fig = prophet_plot(df)

    return fig, round(total_co2,2), round(to_target)
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
st.title('Green Guardians :four_leaf_clover:')
st.markdown('Net0thon 2023')
st.markdown('KFUPM - Dhahran - Saudi Arabia')
'---'
title = "2020 Saudi Arabia's CO2 Emissions"
st.markdown(f"<h1 style='text-align: center; color: white;'>{title}</h1>", unsafe_allow_html=True)
# st.markdown(f"<h1 style='text-align: center; color: white; font-size: medium'>{title}</h1>", unsafe_allow_html=True)

#TODO:
# add source paper
# add co2 data from other paris accord countries
# axis font size

cols = st.columns(3)

# l3_per_yr = cols[0].slider('No. of Liquid Trees:', 0, 1000000, 10000, 100)
options = np.arange(0, 1000000+1, 10000)

l3_per_yr = cols[0].select_slider('No. of Liquid Trees (per year):', options, 10000)
l3_per_yr = int(l3_per_yr)

growth = cols[1].number_input('Growth Rate (%):', 5, 500, 10, 5)
color_by = cols[2].selectbox('Color by:', ['Sector', 'Province', 'Primary Fuel', 'Unit Type'], 0)
# Display KSA CO2 map
# with st.container():
#     st.plotly_chart(co2_map(color_by), use_container_width=True)   # USE COLUMN WIDTH OF CONTAINER


with st.expander("CO2 Emissions Map - 2020"):
    st.plotly_chart(co2_map(color_by), use_container_width=True)   # USE COLUMN WIDTH OF CONTAINER

'---'
st.markdown("<h1 style='text-align: center; color: white;'>Smart Dashboard</h1>", unsafe_allow_html=True)

# l3_per_yr = cols[0].slider('No. of Liquid Trees:', 0, 1e9, 10000, 100)
# growth = cols[1].number_input('Growth Rate (%):', 5, 500, 5, 5)

cols2 = st.columns(4)

# CO2 ML prediction
with st.container():
    growth /= 100
    growth += 1

    fig, total_co2, to_target = co2_ml(l3_per_yr, growth)

    cols2[1].metric('Total CO2 Absorbed', f"{total_co2} MMT")

    cols2[2].metric('Percent from 2030 Target', f"{to_target} %")

    '---'
    st.plotly_chart(fig, use_container_width=True)   # USE COLUMN WIDTH OF CONTAINER

'---'
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



