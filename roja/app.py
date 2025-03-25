import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import seaborn as sns


st.set_page_config(layout="wide")

# Page Title
st.title("THE PANDEMIC: TRENDS IN COVID-19 VARIANTS")

# Sidebar
# st.sidebar.header("Options")
# graph_type = st.sidebar.radio("Select Graph Type:", ("Matplotlib", "Plotly"))

# Data Processing for Part 1
time_series_variants = pd.read_csv('time_series_variants.csv', index_col='Date', parse_dates=True)
variants = time_series_variants.columns[0:]

dfs = []
for variant in variants:
    df = pd.read_csv(f'{variant}.tsv', sep='\t')
    df['Variant'] = variant  # Add variant column
    df['first_seq'] = pd.to_datetime(df['first_seq'])  # Convert to datetime
    df['last_seq'] = pd.to_datetime(df['last_seq'])  
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)

# GRAPH 1: Time Series
st.header("Time Series of Variants")
fig1 = sp.make_subplots(rows=len(variants), cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=variants)

# Add a trace for each variant in its respective subplot
for i, variant in enumerate(variants):
    fig1.add_trace(
        go.Scatter(x=time_series_variants.index, 
                   y=time_series_variants[variant], 
                   mode='lines', 
                   name=variant),
        row=i + 1, col=1
    )

# Update layout for better visualization
fig1.update_layout(
    height=300 * len(variants),  # Adjust height based on the number of subplots
    xaxis_title="Date",
    yaxis_title="Number of Cases per million",
    showlegend=False,  # Disable legend to avoid repetition
    template="plotly"
)

st.plotly_chart(fig1)

# GRAPH 2: Trend, Seasonality, Residuals
st.header("STL Decomposition for Variants")
weekly_data = time_series_variants.resample('W').sum()

fig2 = sp.make_subplots(rows=3, cols=1, shared_xaxes=True, 
                       subplot_titles=("Trend", "Seasonality", "Residuals"))

colors = ['blue', 'green', 'red', 'purple', 'orange']  # Distinct color scheme

for idx, variant in enumerate(weekly_data.columns):  
    stl = STL(weekly_data[variant], seasonal=13)
    res = stl.fit()

    # Add trend plot
    fig2.add_trace(go.Scatter(
        x=weekly_data.index,
        y=res.trend,
        mode='lines',
        name=f"{variant} - Trend",
        line=dict(color=colors[idx % len(colors)])
    ), row=1, col=1)

    # Add seasonality plot
    fig2.add_trace(go.Scatter(
        x=weekly_data.index,
        y=res.seasonal,
        mode='lines',
        name=f"{variant} - Seasonality",
        line=dict(color=colors[idx % len(colors)])
    ), row=2, col=1)

    # Add residuals plot
    fig2.add_trace(go.Scatter(
        x=weekly_data.index,
        y=res.resid,  
        mode='lines',
        name=f"{variant} - Residuals",
        line=dict(color=colors[idx % len(colors)])
    ), row=3, col=1)

# Update layout for better visualization
fig2.update_layout(
    height=900,
    template="plotly_white"
)

# Separate legends for each subplot
fig2.update_layout(
    showlegend=True,
    legend_tracegroupgap=300  # Adjust spacing between legends
)

st.plotly_chart(fig2)

# GRAPH 3: Seasonal Emgergence
st.header("Seasonal Emergence of Variants")
weekly_data['Month'] = weekly_data.index.month
weekly_data['Season'] = weekly_data['Month'].apply(lambda x: 'Winter (Dec-Feb)' if x in [12, 1, 2] else
                                                           'Spring (Mar-May)' if x in [3, 4, 5] else
                                                           'Summer (Jun-Aug)' if x in [6, 7, 8] else
                                                           'Fall (Sep-Nov)')

# Aggregate by season
seasonal_totals = weekly_data.groupby('Season').sum()

# Filter data for India
india_data = data[data['country'] == 'India']

# Aggregate sequences by date and variant for India
india_time_series_variants = india_data.groupby(['first_seq', 'Variant'])['num_seqs'].sum().unstack()

# Ensure the index is a continuous date range
india_full_date_range = pd.date_range(start=india_time_series_variants.index.min(), 
                                      end=india_time_series_variants.index.max(), freq='D')
india_time_series_variants = india_time_series_variants.reindex(india_full_date_range).fillna(0)


# Rename index to 'Date' for clarity
india_time_series_variants.index.name = 'Date'

# Resample data to weekly frequency for smoother analysis
india_weekly_data = india_time_series_variants.resample('W').sum()

# Add 'Season' Column
india_weekly_data['Month'] = india_weekly_data.index.month
india_weekly_data['Season'] = india_weekly_data['Month'].apply(lambda x: 'Winter (Dec-Feb)' if x in [12, 1, 2] else
                                                               'Spring (Mar-May)' if x in [3, 4, 5] else
                                                               'Summer (Jun-Aug)' if x in [6, 7, 8] else
                                                               'Fall (Sep-Nov)')

# Aggregate by season
india_seasonal_totals = india_weekly_data.groupby('Season').sum()

fig3 = go.Figure()

for variant in seasonal_totals.columns:
    fig3.add_trace(go.Bar(
        x=seasonal_totals.index,
        y=seasonal_totals[variant],
        name=variant
    ))

# Update layout for better visualization
fig3.update_layout(
    barmode='stack',
    title="Variant Emergence by Season (Global)",
    xaxis_title="Season",
    yaxis_title="Total Sequences (Log Scale)",
    yaxis_type="log",
    template="plotly_white",
    legend_title="Variants"
)

st.plotly_chart(fig3)

fig4 = go.Figure()

for variant in india_seasonal_totals.columns:
    fig4.add_trace(go.Bar(
        x=india_seasonal_totals.index,
        y=india_seasonal_totals[variant],
        name=variant
    ))

# Update layout for better visualization
fig4.update_layout(
    barmode='stack',
    title="Variant Emergence by Season (India)",
    xaxis_title="Season",
    yaxis_title="Total Sequences (Log Scale)",
    yaxis_type="log",
    template="plotly_white",
    legend_title="Variants"
)

st.plotly_chart(fig4)

# GRAPH 4: Forecasting of Variants
st.header("Forecasting of Variants")

# make dropdown menu for variant
variant = st.selectbox("Select a variant to forecast:", variants)
forecast_steps = st.selectbox("Select the number of weeks to forecast for:", [6, 12, 24])

variant_data = weekly_data[variant].dropna()
forecast_data = pd.read_csv(

# Plot historical data + forecast using Plotly
fig5 = sp.make_subplots(rows=1, cols=2, subplot_titles=[f"{variant} - Historical Data", f"{variant} - Forecast"])

# Add historical data trace
fig5.add_trace(
    go.Scatter(
        x=variant_data.index,
        y=variant_data,
        mode='lines',
        name=f"{variant} - Historical Data"
    ),
    row=1, col=1
)

# Add forecast data trace
fig5.add_trace(
    go.Scatter(
        x=forecast_data.index,
        y=forecast_data,
        mode='lines',
        name=f"{variant} - Forecast",
        line=dict(color='red')
    ),
    row=1, col=2
)

# Update layout for better visualization
fig5.update_layout(
    title=f"ARIMA Forecast for {variant}",
    xaxis_title="Date",
    yaxis_title="Number of Sequences",
    template="plotly_white"
)

st.plotly_chart(fig5)

# GRAPH 5: Mutation Frequency
st.header("Mutation Frequency")
mutation_df = pd.read_csv("mutations.csv", index_col=0)

fig6 = go.Figure(data=go.Heatmap(
                     z=mutation_df.values[::-1],
                     x=mutation_df.columns,
                     y=mutation_df.index[::-1],
                     colorscale='Viridis'))
st.plotly_chart(fig6)

# Text Boxes
st.subheader("Observations")
observation = st.text_area("Enter your analysis here...", "Mutation B has the highest frequency.")

st.subheader("Summary")
st.write("This dashboard helps track mutation frequencies and analyze trends.")

# Display user input
if observation:
    st.write("### Your Notes:")
    st.write(observation)
