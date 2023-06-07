# import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import glob, math
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

def millify(n):
    # millnames = ['', ' Thousand', ' Million', ' Billion', ' Trillion']
    millnames = ['', ' K', ' M', ' B', ' T']
    n = float(n)
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))

    return '{:.0f}{}'.format(n / 10 ** (3 * millidx), millnames[millidx])

def co2_map(color_by):
    # Read files
    fnames = glob.glob('./data/by_sector/*.csv')

    df_list = []
    for f in fnames:
        df_list.append(pd.read_csv(f, index_col=False))

    df = pd.concat(df_list)

    order = {'Sector': ['Electricity', 'Desalination', 'Petrochemicals', 'Refinery', 'Cement', 'Steel']}
    colors = ['red', 'blue', 'yellow', 'green', 'cyan', 'purple', 'orange']

    fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", hover_name="City",
                            hover_data=["Sector", "CO2 emission (Mton/yr)"],
                            size="CO2 emission (Mton/yr)",
                            size_max=30,
                            category_orders=order,
                            color=color_by,
                            color_discrete_sequence=colors,
                            zoom=3,
                            # width=450,
                            height=500)
    fig.update_layout(
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
            n_changepoints=len(ds)//4,
            changepoint_range=.92,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=None,
            seasonality_mode='additive',
            seasonality_prior_scale= 1.25, #0.05,  # yhat zigzag
            holidays_prior_scale=0,
            changepoint_prior_scale=1.25, #  .25,  # yhat slope, largefr == steeper
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
                             hovertemplate='<i>Pop.</i>: %{y:d}' +
                                           '<br><i>Year</i>: %{x|%Y}<br><extra></extra>',
                             name='Population',
                             showlegend=True),
                  secondary_y=True)

    # plot population Forecast
    fig.add_trace(go.Scatter(mode='lines', x=x, y=d.yhat_x,
                             line=dict(color='magenta', width=2, dash='dash'),
                             hovertemplate='<i>Pop. Forecast</i>: %{y:d}' +
                                           '<br><i>Year</i>: %{x|%Y}<br><extra></extra>',
                             name='Pop. Forecast',
                             showlegend=True),
                  secondary_y=True)

    # plot (CO2 emissions) Actual
    fig.add_trace(go.Scatter(mode='lines', x=x, y=d.co2_mt,
                             line=dict(color=color, width=2),
                             hovertemplate='<i>CO2</i>: %{y:.1f} Million Tons' +
                                           '<br><i>Year</i>: %{x|%Y}<br><extra></extra>',
                             name='CO2',
                             showlegend=True),
                  secondary_y=False)

    # plot (CO2 emissions) Forecast
    fig.add_trace(go.Scatter(mode='lines', x=x, y=d.yhat_y,
                             line=dict(color=color, width=2, dash='dash'),
                             hovertemplate='<i>CO2</i>: %{y:.1f} Million Tons' +
                                           '<br><i>Year</i>: %{x|%Y}<br><extra></extra>',
                             name='CO2 Forecast',
                             showlegend=True),
                  secondary_y=False)

    # plot (CO2 abatment)
    fig.add_trace(go.Scatter(mode='lines', x=x, y=d.abate,
                             line=dict(color='blue', width=2, dash='dash'),
                             hovertemplate='<i>CO2</i>: %{y:.1f} Million Tons' +
                                           '<br><i>Year</i>: %{x|%Y}<br><extra></extra>',
                             name='CO2 Reduction',
                             showlegend=True),
                  secondary_y=False)

    # # plot (CO2 abatment) forecast
    # fig.add_trace(go.Scatter(mode='lines', x=x, y=d.abate2,
    #                          line=dict(color='blue', width=2, dash='dash'),
    #                          hovertemplate='<i>CO2</i>: %{y:.2f} Million Ton' +
    #                                        '<br><i>Year</i>: %{x|%Y}<br><extra></extra>',
    #                          name='Abate. Estimate',
    #                          showlegend=True),
    #               secondary_y=False)

    fig.add_hline(y=278, line_width=3, line_dash="dash", line_color="yellow", annotation_text="2030 278MM Goal",
                  annotation_position="top left",
                  annotation_font_size=14,
                  )

    fig.add_hline(y=d.yhat_y.iloc[-1], line_width=3, line_dash="dash", line_color="cyan",
                  annotation_text="2060 Net-zero Goal",
                  annotation_position="top left",
                  annotation_font_size=14,
                  )

    fig.update_layout(
        hovermode="closest",
        hoverlabel=dict(align="left", bgcolor="rgba(0,0,0,0)"),
        #         template="plotly_dark",
        margin=dict(t=0, b=0, l=0, r=0),

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        yaxis=dict(zeroline= False, showgrid=False, tickfont = dict(size=18),
                   title={"font": dict(size=20), "text": "CO2 (Million Ton)", "standoff": 10}),
        yaxis2=dict(zeroline=False, showgrid=False, range=[0, 50e6], tickfont = dict(size=18),
                    title={"font": dict(size=20), "text": "Population", "standoff": 10}),
        xaxis=dict(showline=False, tickfont = dict(size=18),)
    )
    return fig


def co2_ml(n_co2_wells, co2_rate, n_geo_wells, power_rate_y, co2_saved_yr, n_l3_y, l3_rate_mty):

    fnames = glob.glob('./data/*.csv')
    n = 15  # future forecast years

    # Read csv of existing co2 efforts
    df_list = []
    for f in fnames:
        df = pd.read_csv(f).T.dropna(subset=[0])
        df_list.append(df)

    df = pd.concat(df_list, axis=1)
    df.rename(columns={0: 'co2_kt', 1: 'pop',
                       2: 'utmn_eor', 3: 'sabic', 4: 'mangrove'},
              inplace=True)

    # unit conversions
    df['co2_mt'] = df.loc[:, 'co2_kt'] / 1000 # kilo ton to million tons
    df.mangrove *= 1e3 * 1e4 * 20 * 1e-3 * 1e-6 # kilo hectar to mty (assume 20 kg/m2/yr rate)

    # convert index to datetime
    df.index = [f"{y}-12-31" for y in df.index]
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')

    # Forecast fb prophet
    cols = ['co2_mt', 'pop']   # these columns will be forcasted
    df = annual_prophecy(df, cols, forecast_period=n)

    # compute impact of scenarios for dashboard
    annual_impact = np.arange(n) + 1.0

    # co2 wells
    # total_co2_wells = annual_impact * n_co2_wells
    # co2_wells_impact = total_co2_wells * co2_rate
    df['cwells'] = 0
    df['cwells'].iloc[-n:] = annual_impact * n_co2_wells
    df['cwells_co2'] = df.cwells * co2_rate
    # geothermal wells
    # total_geo_wells = annual_impact * n_geo_wells
    # geo_wells_impact = total_geo_wells * co2_saved_yr
    df['gwells'] = 0
    df['gwells'].iloc[-n:] = annual_impact * n_geo_wells
    df['gwells_co2'] = df.gwells * co2_saved_yr
    df['power_gen_y'] = df.gwells * power_rate_y * 1e-6  # TWh
    # power_generated_y = total_geo_wells * power_rate_y * 1e-6 # TWh

    # Liquid 3
    # total_l3 = annual_impact * n_l3_y
    # l3_impact = total_l3 * l3_rate_mty
    df['l3'] = 0
    df['l3'].iloc[-n:] = annual_impact * n_l3_y
    df['l3_co2'] = df.l3 * l3_rate_mty

    # include other kingdom efforts
    # df['abate'] = df.utmn_eor + df.sabic + df.mangrove

    df.mangrove.pad(inplace=True)
    df.mangrove.iloc[-n:] += (annual_impact * 0.5)     # assume mangroves planting rate is steady
    df.sabic.pad(inplace=True)
    df.utmn_eor.pad(inplace=True)

    # Combined future impact
    df['abate'] = df.utmn_eor + df.sabic + df.mangrove + df.cwells_co2 + df.gwells_co2 + df.l3_co2
    # df['abate2'] = df.abate.pad()

    # df.abate2.iloc[-n:] += l3_impact + co2_wells_impact + geo_wells_impact
    # df.abate.iloc[-n:] += l3_impact + co2_wells_impact + geo_wells_impact

    # total CO2 absorbed
    # total_co2 = sum(l3_impact + co2_wells_impact + geo_wells_impact) * 1e-3   # billion tons

    # CO2 absorption rate by 2030
    dt = pd.to_datetime(['2030','2031'])
    co2_2030 = df.abate.loc[(df.index >= dt[0]) & (df.index <= dt[-1])].values[0]

    # get % to meet target
    to_target = (co2_2030/278)*100

    # get last forcast year
    year_end = df.index[-1].strftime('%Y')

    df.drop(columns=['trend_x', 'trend_lower_x', 'trend_upper_x',
                     'trend_y', 'trend_lower_y', 'trend_upper_y',
                     'yhat_lower_x', 'yhat_upper_x', 'yhat_lower_y', 'yhat_upper_y'], errors='ignore', inplace=True)

    fig = prophet_plot(df)

    df

    return fig, df, round(to_target), year_end


def make_pie(df):

    last_row = df.loc[:,['cwells_co2', 'gwells_co2','l3_co2','mangrove','sabic','utmn_eor']].iloc[-1,:].to_list()
    methods=['CO2 Wells', 'Geothermal Wells','Liquid Trees', 'Mangroves','SABIC','EOR']

    arr = np.array([methods, last_row]).T
    temp_df = pd.DataFrame(arr, columns=['Method', 'CO2'])

    fig = px.pie(temp_df, values='CO2', names='Method', hole=0.5)
    fig.update_traces(textposition='inside', textinfo='percent+label',textfont_size=18)

    fig.update_layout(
        template="plotly_dark",
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # title={'text': 'TOP 10 HOLDERS', "font": dict(size=24)},
    )
    return fig
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
# st.title('Green Guardians :four_leaf_clover:')
st.title('Sustainability Dashboard :four_leaf_clover:')
st.markdown("Build scenarios to accelerate the decarbonization of the Kingdom of Saudi Arabia")
st.markdown('Net0thon 2023 - Dhahran - Saudi Arabia')
'---'

#TODO:
# add source paper
# add co2 data from other paris accord countries
# axis font size

################################## CONTROL PANEL
st.markdown("<h1 style='text-align: center; color: white; font-size: medium'>CONTROL PANEL</h1>", unsafe_allow_html=True)
cols = st.columns([6,1,6], gap='small')
# CO2 sequestration wells:
n_co2_wells = cols[0].number_input('No. of CO2 sequestration wells:',min_value=0,max_value=None,value=5,step=1,
                                   help="Number of CO2 sequestration wells drilled annually.")
co2_rate = cols[-1].slider('CO2 sequestration rate (Mt/yr):',min_value=0.5,max_value=50.0,value=5.0,step=0.5,
                          help="Average CO2 sequestration rate per well in Million tons per year (Mt/yr). "
                               "A typical CO2 storage well has a rate of 10 Mt/yr, while an enhanced oil recovery well"
                               " has a rate of 1-5 Mt/yr.")
# geothermal wells:
n_geo_wells = cols[0].number_input('No. of geothermal wells:',min_value=0,max_value=None,value=5,step=1,
                                   help="Number of geothermal wells drilled annually.")
power_rate_d = cols[-1].slider('Power output (MWh/day):',min_value=0.5,max_value=50.0,value=1.0,step=0.5,
                          help="Average power output per well in mega watt hours (MWh) per day. This rate will be "
                               "multiplied by 365, assuming the well is running 24/7.")
power_rate_y = power_rate_d * 365 * 1000 # kwh/yr
co2_saved_yr = power_rate_y * 0.65 * 1e-3 * 1e-6  # million tons CO2 annually

# Liquid Trees:
n_l3 = cols[0].number_input('No. of Liquid Trees:', 0, None, 100, 100,help="Number of liquid trees installed annually")
l3_rate_kgy = cols[-1].slider('CO2 absorption rate (Kg/yr):',min_value=10,max_value=1000,value=100,step=10,
                        help="Average CO2 absorption (abatement) rate per tree in kilograms per year")
l3_rate_mty = l3_rate_kgy * 1e-3 * 1e-6 # million tons co2 annually

# typical emission rate is within 0.37–0.48 kg CO2 per kWh
# 282 Tera-Watt-hours generated by power plants corresponding to CO2 emissions of 183 million tons annually;
# Thus, the resulting average emission rate is about 0.65 kgCO2 per kWh.
# terra watt is 10^2, mega is 10^6
# For example, wells that are used to store CO2 in deep saline aquifers can have injection rates of up to 10 Mt/yr,
# while wells that are used to store CO2 in depleted oil and gas reservoirs typically have injection rates of 1-5 Mt/yr.

################################## Smart DASHBOARD
'---'
with st.container():

    # CO2 ML prediction
    fig, df, to_target, year_end = co2_ml(
        n_co2_wells, co2_rate, n_geo_wells, power_rate_y, co2_saved_yr,  n_l3, l3_rate_mty)

    # METRICS TITLE
    st.markdown(f"<h1 style='text-align: center; color: white; font-size: medium'>{year_end} SUSTAINABILITY DASHBOARD</h1>",
                unsafe_allow_html=True)

    # METRIC COLUMNS
    cols2 = st.columns([1, 5, 5, 5, 1], gap='small')  ########## METRICS COLUMNS

    # COMPUTE METRICS
    total_co2 = df.loc[:,['cwells_co2', 'gwells_co2','l3_co2']].sum().sum() * 1e-3    # billion tons

    # DISPLAY METRICS
    cols2[1].metric('Total CO2 Wells Drilled', f"{millify(df.cwells[-1])}")
    cols2[2].metric('Total Geothermal Wells Drilled', f"{millify(df.gwells[-1])}")
    cols2[3].metric('Total Liquid Trees Installed', f"{millify(df.l3[-1])}")
    cols2[1].metric('Total CO2 Absorbed', f"{total_co2:.1f} BT")
    cols2[2].metric('Total Power Generated', f"{df.power_gen_y[-1]:.1f} TWh")
    cols2[3].metric('Percent from 2030 Target', f"{to_target} %")
    '---'

    # CHART TITLE
    title = "Saudi Arabia's CO2 & Population Forecast"
    st.markdown(f"<h1 style='text-align: center; color: white; font-size: medium'>{title}</h1>",
                unsafe_allow_html=True)
    # PLOT DASHBOARD
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("source: [World Bank](https://data.worldbank.org/)")


################################## CO2 MAP
'---'
title = "2020 Saudi Arabia's CO2 Emissions"
st.markdown(f"<h1 style='text-align: center; color: white; font-size: medium'>{title}</h1>", unsafe_allow_html=True)
# st.markdown(f"<h1 style='text-align: center; color: white; font-size: medium'>{title}</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs([f"CO2 Reduction Effectiveness - {year_end} ", "CO2 Emissions Map - 2020"])

with tab1:
    st.header(f"Contribution to Total CO2 Reduction by {year_end}")
    # st.bar_chart(data=df, y=['cwells_co2', 'gwells_co2','l3_co2','mangrove','sabic','utmn_eor'])
    st.plotly_chart(make_pie(df), use_container_width=True)

with tab2:
    st.header("2020 Saudi Arabia's CO2 Emissions")
    color_by = st.selectbox('Color by:', ['Sector', 'Province', 'Primary Fuel', 'Unit Type'], 0)
    st.plotly_chart(co2_map(color_by), use_container_width=True)  # USE COLUMN WIDTH OF CONTAINER
    st.markdown("source: [Rowaihy et al., 2022](https://www.sciencedirect.com/science/article/pii/S2590174522001222)")

# with st.expander("CO2 Emissions Map - 2020", True):
#     color_by = st.selectbox('Color by:', ['Sector', 'Province', 'Primary Fuel', 'Unit Type'], 0)
#     st.plotly_chart(co2_map(color_by), use_container_width=True)   # USE COLUMN WIDTH OF CONTAINER
#     st.markdown("source: [Rowaihy et al., 2022](https://www.sciencedirect.com/science/article/pii/S2590174522001222)")

'---'

with st.expander("Did you know?"):
    '- The electricity sector is the largest consumer of domestic oil and gas in KSA, ' \
    'where electricity generation is growing 6.3% annually.'
    '- In 2020, the kingdom generated 282 TWh to meet consumer power demand.'
