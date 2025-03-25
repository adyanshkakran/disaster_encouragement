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

st.header("Introduction")
st.write("ello apan k aproject hai")

# Data Processing
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

st.markdown("""
    <style>
    .st-expander {
        font-size: 20px !important;  /* Adjust size as needed */
        font-weight: bold !important; /* Make it bold */
    }
    </style>
""", unsafe_allow_html=True)

# 📈 Time Series of Variants
with st.expander("📈 Time Series of Variants", expanded=False):
# st.header("📈 Time Series of Variants")
    summary = """Pmdarima  \n\n\n\n\n (originally pyramid-arima, for the anagram of 'py' + 'arima') is a statistical library designed to fill the void in Python's time series analysis capabilities. This includes
    The equivalent of R's auto.arima functionality
    A collection of statistical tests of stationarity and seasonality
    Time series utilities, such as differencing and inverse differencing
    Numerous endogenous and exogenous transformers and featurizers, including Box-Cox and Fourier transformations
    Seasonal time series decompositions
    Cross-validation utilities
    A rich collection of built-in time series datasets for prototyping and examples
    Scikit-learn-esque pipelines to consolidate your estimators and promote productionization
    Pmdarima wraps statsmodels under the hood, but is designed with an interface that's familiar to users coming from a scikit-learn background."""

    # Create a 2x3 grid using st.columns()
    cols = st.columns(2)  # 3 columns in the first row

    # First box (Summary Text) in the first column
    with cols[0]:  
        # st.markdown("### Summary")
        st.write(summary)
        
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    # Add first 2 variant plots in remaining columns of the first row
    for i, variant in enumerate(variants[:1]):  
        with cols[i + 1]:  
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_series_variants.index, 
                y=time_series_variants[variant], 
                mode='lines', 
                name=variant,
                line=dict(color=colors[0])  # Add color
            ))
            fig.update_layout(
                title=variant,
                xaxis_title="Date",
                yaxis_title="Cases per Million",
                template="plotly_white"
            )
            st.plotly_chart(fig)

    # Second row (create 3 more columns)
    cols2 = st.columns(2)

    # Add remaining 3 variant plots in the second row
    for i, variant in enumerate(variants[1:3]):  
        with cols2[i]:  
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_series_variants.index, 
                y=time_series_variants[variant], 
                mode='lines', 
                name=variant,
                line = dict(color=colors[1:3][i])  # Add color
            ))
            fig.update_layout(
                title=variant,
                xaxis_title="Date",
                yaxis_title="Cases per Million",
                template="plotly_white"
            )
            st.plotly_chart(fig)
            
            
    # Third row (create 3 more columns)
    cols3 = st.columns(2)

    # Add remaining 3 variant plots in the third row
    for i, variant in enumerate(variants[3:]):
        with cols3[i]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(     
                x=time_series_variants.index,
                y=time_series_variants[variant],
                mode='lines',
                name=variant,
                line = dict(color=colors[3:][i])  
            ))
            fig.update_layout(
                title=variant,
                xaxis_title="Date",
                yaxis_title="Cases per Million",
                template="plotly_white"
            )
            st.plotly_chart(fig)
                        

# 📊 STL Decomposition
with st.expander("📊 STL Decomposition for Variants", expanded=False):
    weekly_data = time_series_variants.resample('W').sum()
    
    # Create 2x2 columns
    cols = st.columns(2)

    # First column (Row 1): Explanation Text
    with cols[0]:  
        # st.markdown("### Understanding STL Decomposition")
        st.write(
            "- **Trend**: The long-term movement in the time series.\n"
            "- **Seasonality**: Recurring patterns or cycles at regular intervals.\n"
            "- **Residuals**: The remaining variation after removing trend and seasonality."
        )
    
    # Second column (Row 1): Trend Plot
    with cols[1]:  
        fig_trend = go.Figure()
        colors = ['blue', 'green', 'red', 'purple', 'orange']

        for idx, variant in enumerate(weekly_data.columns):  
            stl = STL(weekly_data[variant], seasonal=13)
            res = stl.fit()
            fig_trend.add_trace(go.Scatter(
                x=weekly_data.index, 
                y=res.trend, 
                mode='lines', 
                name=f"{variant} - Trend",
                line=dict(color=colors[idx % len(colors)])
            ))

        fig_trend.update_layout(
            title="Trend",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # New row for Seasonality and Residuals
    cols = st.columns(2)

    # First column (Row 2): Seasonality Plot
    with cols[0]:  
        fig_seasonality = go.Figure()
        for idx, variant in enumerate(weekly_data.columns):  
            stl = STL(weekly_data[variant], seasonal=13)
            res = stl.fit()
            fig_seasonality.add_trace(go.Scatter(
                x=weekly_data.index, 
                y=res.seasonal, 
                mode='lines', 
                name=f"{variant} - Seasonality",
                line=dict(color=colors[idx % len(colors)])
            ))

        fig_seasonality.update_layout(
            title="Seasonality",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white"
        )
        st.plotly_chart(fig_seasonality, use_container_width=True)

    # Second column (Row 2): Residuals Plot
    with cols[1]:  
        fig_residuals = go.Figure()
        for idx, variant in enumerate(weekly_data.columns):  
            stl = STL(weekly_data[variant], seasonal=13)
            res = stl.fit()
            fig_residuals.add_trace(go.Scatter(
                x=weekly_data.index, 
                y=res.resid, 
                mode='lines', 
                name=f"{variant} - Residuals",
                line=dict(color=colors[idx % len(colors)])
            ))

        fig_residuals.update_layout(
            title="Residuals",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white"
        )
        st.plotly_chart(fig_residuals, use_container_width=True)


# 🌍 Seasonal Emergence
with st.expander("🌍 Seasonal Emergence of Variants", expanded=False):
    st.write(
        "YOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOo"
    )
    weekly_data['Month'] = weekly_data.index.month
    weekly_data['Season'] = weekly_data['Month'].apply(lambda x: 'Winter (Dec-Feb)' if x in [12, 1, 2] else
                                                               'Spring (Mar-May)' if x in [3, 4, 5] else
                                                               'Summer (Jun-Aug)' if x in [6, 7, 8] else
                                                               'Fall (Sep-Nov)')
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
    for variant in seasonal_totals.columns[:5]:
        fig3.add_trace(go.Bar(x=seasonal_totals.index, y=seasonal_totals[variant], name=variant))

    fig3.update_layout(barmode='stack', title="Variant Emergence by Season (Global)",
                       xaxis_title="Season", yaxis_title="Total Sequences (Log Scale)",
                       yaxis_type="log", template="plotly_white", legend_title="Variants")
    
    # st.plotly_chart(fig3)
    
    fig4 = go.Figure()

    for variant in india_seasonal_totals.columns[:5]:
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

    # st.plotly_chart(fig4)
    
    # plot fig3 and fig 4 side by side busing st.col
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig3)
    with col2:
        st.plotly_chart(fig4)

# 🔮 Forecasting of Variants
with st.expander("🔮 Forecasting of Variants", expanded=False):
    variant = st.selectbox("Select a variant to forecast:", variants)
    forecast_steps = st.selectbox("Select the number of weeks to forecast for:", [6, 12, 24])

    variant_data = weekly_data[variant].dropna()
    forecast_data = pd.read_csv(f"{variant}_forecast_{forecast_steps}.csv", index_col='Date', parse_dates=True)['predicted_mean']

    fig5 = sp.make_subplots(rows=1, cols=2, subplot_titles=[f"{variant} - Historical Data", f"{variant} - Forecast"])
    fig5.add_trace(go.Scatter(x=variant_data.index, y=variant_data, mode='lines', name=f"{variant} - Historical Data"), row=1, col=1)
    fig5.add_trace(go.Scatter(x=forecast_data.index, y=forecast_data, mode='lines', name=f"{variant} - Forecast",
                              line=dict(color='red')), row=1, col=2)

    fig5.update_layout(title=f"ARIMA Forecast for {variant}", xaxis_title="Date", yaxis_title="Number of Sequences",
                       template="plotly_white")
    st.plotly_chart(fig5)

# 🧬 Mutation Frequency
with st.expander("🧬 Mutation Frequency", expanded=False):
    # st.markdown("### Understanding Mutation Frequency")
    st.write(
        "This heatmap visualizes the prevalence of different mutations across COVID-19 variants. "
        "Darker shades indicate higher mutation frequencies, helping to track variant evolution over time."
    )
    
    mutation_df = pd.read_csv("mutations.csv", index_col=0)
    
    # Create a smaller heatmap
    fig6 = go.Figure(data=go.Heatmap(
        z=mutation_df.values[::-1], 
        x=mutation_df.columns, 
        y=mutation_df.index[::-1], 
        colorscale='Viridis',
        colorbar=dict(title="Frequency")  # Add color bar label
    ))

    fig6.update_layout(
        autosize=False, 
        width=700,  # Reduce width for better readability
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Center the plot using Streamlit's container
    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column widths as needed
    with col2:
        st.plotly_chart(fig6, use_container_width=True)  # Center the plot and allow it to stretch within the middle column

