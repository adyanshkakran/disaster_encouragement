import streamlit as st
import matplotlib.pyplot as plt
import plotly.subplots as sp
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import re
import time
from matplotlib import animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# import PillowWriter
from matplotlib.animation import PillowWriter

import sys
sys.path.append('../simulation/code')

from run_simulations import run_simulation
# import pmdarima as pm
# import seaborn as sns

st.set_page_config(layout="wide")

# Page Title
st.title("THE PANDEMIC: TRENDS IN COVID-19 VARIANTS")

st.markdown(""" 
## Introduction
The **COVID-19 pandemic** was one of the most significant global health crises in modern history, caused by the **SARS-CoV-2 virus**. First identified in late 2019, the virus spread rapidly, leading to widespread illness, healthcare strain, and unprecedented public health responses.
            """)
col1, col2 = st.columns([3, 2])  # Adjust column widths as needed

with col1:
    st.write("""   
    ### **Understanding COVID-19 Variants**  
    Viruses constantly evolve, and **COVID-19 variants** emerged as the virus spread worldwide. Some of these changes had **major public health implications**, leading to the classification of variants into three categories:  

    - **Variant Under Monitoring (VUM)** – A variant that requires close tracking due to potential risks.  
    - **Variant of Interest (VOI)** – A variant with genetic changes that may affect its spread, severity, or treatment.  
    - **Variant of Concern (VOC)** – A variant that significantly impacts disease severity, vaccine effectiveness, or healthcare systems.

    ### **What This Dashboard Offers**  
    This interactive dashboard provides a **quick and clear overview** of COVID-19 and its variants, tracking their prevalence, mutations, and seasonal patterns. 

    The variants we’ve focused on in this dashboard are:  

    - **Alpha (B.1.1.7)**  
    - **Beta (B.1.351)** 
    - **Gamma (B.1.1.28.1)** 
    - **Delta (B.1.617.2)**  
    - **Omicron (B.1.1.529)** 

    These variants played a key role in shaping the course of the pandemic, influencing public health responses and vaccine development.
    """)

with col2:
    st.image("intro.jpg", caption="COVID-19 Variants Emergence", use_container_width=True)

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

# st.markdown("""
#     <style>
#     .st-expander {
#         font-size: 20px !important;  /* Adjust size as needed */
#         font-weight: bold !important; /* Make it bold */
#     }
#     </style>
# """, unsafe_allow_html=True)

# 📈 Time Series of Variants
with st.expander("📈 Time Series of Variants", expanded=False):
# st.header("📈 Time Series of Variants")

    # Create a 2x3 grid using st.columns()
    cols = st.columns(2)  # 3 columns in the first row

    # First box (Summary Text) in the first column
    with cols[0]:  
        st.markdown("""
    # Time Series of COVID-19 Variants
    
    These graphs present the global time series vs cases per million of COVID-19 variants over time. Each subplot represents a specific variant (**Beta, Alpha, Gamma, Delta, Omicron**).
    ### **Key Observations:**  
    - **Alpha & Beta (Early 2021):** These variants showed early peaks but declined as new variants emerged.  
    - **Gamma & Delta (Mid-2021):** Delta exhibited a strong presence, outcompeting earlier variants due to its higher transmissibility.  
    - **Omicron (Late 2021 - 2022):** A sharp increase in Omicron sequences highlights its rapid spread and dominance, surpassing previous variants.  

    This visualization shows how different variants **rose and fell over time**, correlating with major COVID-19 waves worldwide.
                    """)
        
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    # Add first 2 variant plots in remaining columns of the first row
    for i, variant in enumerate(variants[:1]):  
        with cols[i + 1]:  
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_series_variants.index, 
                y=time_series_variants[variant] // 199, 
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
                y=time_series_variants[variant] // 199, 
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
                y=time_series_variants[variant] // 199,
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
        st.markdown("""
                    # **STL Decomposition for Variants**  

STL (Seasonal-Trend Decomposition using LOESS) helps break down time series data into three components: **Trend, Seasonality, and Residuals**, providing insights into COVID-19 variant spread.  

1. **Trend:** Captures long-term movement in variant prevalence, showing whether a variant is gaining dominance and how interventions affect its spread.  
2. **Seasonality:** Identifies recurring patterns influenced by weather, human behavior, and epidemiological cycles, helping distinguish periodic spikes from true variant growth.  
3. **Residuals:** Represents unexplained variations due to sudden mutations, policy changes, or reporting inconsistencies, signaling unpredictable shifts in variant spread.  

### **Why Use STL Decomposition?**  
- Differentiates between sustained growth and short-term fluctuations.  
- Highlights seasonal patterns that impact outbreaks.  
- Detects unexpected shifts, possibly indicating new variants or external influences.
                    
                    """)
    
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
    st.markdown("""

# Variant Emergence by Season (Global vs. India)  

### **Key Insights:**
1. **Delta (21A.Delta)** originated in India and peaked early (Spring 2021), while globally, it became dominant later in Summer and Fall (Jun-Nov).  
2. **Omicron (21K.Omicron)** dominated globally in Fall (Sep-Nov), but India saw a relatively lower surge compared to the world.  
3. **Alpha (20I.Alpha.V1)** was significant globally in Winter (Dec-Feb), but its impact in India was lower due to Delta’s early spread.  
4. **Beta (20H.Beta.V2)** had a continuous presence, especially in Winter, showing its persistence in India and globally.  
5. **Gamma (20J.Gamma.V3)** was a global concern but had minimal presence in India.  """)
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
    st.markdown("""
    # **ARIMA Forecast for COVID-19 Variants**  
    Here we display an **ARIMA (AutoRegressive Integrated Moving Average) forecast** for the COVID-19 variant of your choice. The left graph shows historical case counts over time, while the right graph presents the projected case trends for the next **n** chosen weeks.  

    ### **How ARIMA Works**  
    ARIMA is a time series forecasting model that combines three key components:  
    - **AutoRegression (AR):** Uses past values to predict future values.  
    - **Differencing (I):** Removes trends to make the data stationary.  
    - **Moving Average (MA):** Accounts for past forecast errors to improve predictions.  

    ARIMA is effective in capturing temporal patterns and making short-term forecasts, as seen in the increasing trend in the right graph. However, it assumes that future trends follow past patterns, so unexpected external factors (e.g., new mutations, policy changes) may impact real-world outcomes.
                """)
    variant = st.selectbox("Select a variant to forecast:", variants)
    forecast_steps = st.selectbox("Select the number of weeks to forecast for:", [6, 12, 24])

    variant_data = weekly_data[variant].dropna()
    forecast_data = pd.read_csv(f"{variant}_forecast_{forecast_steps}.csv", index_col='Date', parse_dates=True)['predicted_mean']

    fig5 = sp.make_subplots(rows=1, cols=2, subplot_titles=[f"{variant} - Historical Data", f"{variant} - Forecast"])
    fig5.add_trace(go.Scatter(x=variant_data.index, y=variant_data // 199, mode='lines', name=f"{variant} - Historical Data"), row=1, col=1)
    fig5.add_trace(go.Scatter(x=forecast_data.index, y=forecast_data // 199, mode='lines', name=f"{variant} - Forecast",
                              line=dict(color='red')), row=1, col=2)

    fig5.update_layout(title=f"ARIMA Forecast for {variant}", xaxis_title="Date", yaxis_title="Number of Cases (per million people)",
                       template="plotly_white")
    st.plotly_chart(fig5)

# 🧬 Mutation Frequency
with st.expander("🧬 Mutation Frequency", expanded=False):
    # st.markdown("### Understanding Mutation Frequency")
    st.markdown("""

# Mutations and Variants Explained 
The Centers for Disease Control (CDC) defines a mutation as a single change in a virus’ genome or genetic code. Viruses like COVID-19 constantly mutate as they replicate in human cells. Most mutations are minor, but some can give the virus an advantage—making it spread faster or evade immunity. When this happens, a new **variant** is formed.   Scientists track these changes by mapping the virus’s genetic code, allowing them to observe COVID-19’s evolution in real time. To simplify variant tracking, the **World Health Organization (WHO)** named them after Greek letters (Alpha, Beta, Delta, Omicron, etc.).  
Think of the virus as a tree—COVID-19 is the trunk, and variants are its growing branches. Some, like Omicron, have mutations that increase transmissibility but cause milder disease. By studying these changes, scientists can assess risks and predict how future variants might behave.
                """)
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
        title="Mutation Frequency Heatmap",
        yaxis_title="Mutations",
        xaxis_title="Variants",
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Center the plot using Streamlit's container
    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column widths as needed
    with col2:
        st.plotly_chart(fig6, use_container_width=True)  # Center the plot and allow it to stretch within the middle column
        
        
    st.markdown("""
                
### **Key Takeaways from the Heatmap**  
- **Unique Mutation Profiles:** Each variant (Alpha, Beta, Gamma, Delta, Omicron) has distinct dominant mutations, though some (e.g., **S:N501Y, S:P681H**) appear across multiple variants due to convergent evolution.  
- **Mutation Impact on Spread:** Highly prevalent mutations (darker shades) often drive **higher transmissibility and immune escape**. Example: **Delta (S:P681R) → Faster Spread**, **Omicron (Multiple Mutations) → Higher Reinfection Rates**.  
- **Variants Replacing Each Other:** Each new dominant variant had mutations that gave it a survival edge—either **faster transmission or better immune evasion**. **Delta replaced Alpha**, and **Omicron outcompeted Delta** due to stronger immune escape.  
- **Mutation Accumulation Over Time:** Some mutations persisted across variants, while others evolved further. **Omicron retained earlier mutations while introducing new ones.**  
- **Epidemiological Waves & Variant Evolution:** Major infection waves align with dominant variants:  
    - **Alpha → Beta → Gamma** (Early 2021)  
    - **Delta Surge** (Mid-to-Late 2021)  
    - **Omicron Dominance** (Late 2021 - 2022)  
- **Conclusion:** Mutations **directly influenced variant success** and led to successive COVID-19 outbreaks. The heatmap visually captures how **mutations shaped variant evolution** over time.
                """)
    
def read_result_file(file_name):
    with open(file_name, "r") as file:
        content = file.read()

    # Extract values using regular expressions
    deaths_match = re.search(r"Total number of deaths:\s+([\d\.]+)", content)
    icu_match = re.search(r"Days with exceeded ICU bed capacity:\s+([\d\.]+)", content)
    duration_match = re.search(r"Duration of pandemic \(in days\):\s+([\d\.]+)", content)
    
    # Store values (convert to float)
    total_deaths = int(float(deaths_match.group(1))) if deaths_match else None
    icu_exceeded_days = int(float(icu_match.group(1))) if icu_match else None
    pandemic_duration = int(float(duration_match.group(1))) if duration_match else None
    return total_deaths, icu_exceeded_days, pandemic_duration


# Run animation dynamically
# def run_animation(animation_obj, plot_placeholder):
#     for frame in range(0, animation_obj.steps, 2):  # Skip every other frame for smoother animation
#         update_data = animation_obj.update(frame)
#         animation_obj.fig.data[0].x = update_data["x"]
#         animation_obj.fig.data[0].y = update_data["y"]
#         animation_obj.fig.data[0].marker = update_data["marker"]
#         animation_obj.fig.layout.annotations = update_data["annotations"]
        
#         # Update the figure in Streamlit
#         plot_placeholder.pyplot(animation_obj.fig, use_container_width=True)
#         time.sleep(0.2)  # Increase sleep duration for smoother rendering

def run_animation(animation_obj, fig, ax):
    def update(frame):
        animation_obj.update(frame)
        ax.clear()
        ax.scatter(animation_obj.positions[:, 0], animation_obj.positions[:, 1], 
                   c=[animation_obj.colors[int(s)] for s in animation_obj.states])
        ax.set_title(f"Step {frame + 1} / {animation_obj.steps}")

    ani = animation.FuncAnimation(fig, update, frames=animation_obj.steps, interval=100, repeat=False)
    return ani

with st.expander("🦠 SEI³RD Pandemic Simulator", expanded=False):
    # st.write("ello apan ka simulation")
    
    eqs = [
        r"\frac{dS}{dt} = - \sum_{\ell=1}^{K} \left( \beta^{\text{asym}}_{\ell} I^{\text{asym}}_{\ell} + \beta^{\text{sym}}_{\ell} I^{\text{sym}}_{\ell} + \beta^{\text{sev}}_{\ell} I^{\text{sev}}_{\ell} \right) S",
        r"\frac{dE}{dt} = \sum_{\ell=1}^{K} \left( \beta^{\text{asym}}_{\ell} I^{\text{asym}}_{\ell} + \beta^{\text{sym}}_{\ell} I^{\text{sym}}_{\ell} + \beta^{\text{sev}}_{\ell} I^{\text{sev}}_{\ell} \right) S - \varepsilon E",
        r"\frac{dI^{\text{asym}}}{dt} = \eta \varepsilon E - \gamma^{\text{asym}} I^{\text{asym}}",
        r"\frac{dI^{\text{sym}}}{dt} = (1 - \eta)(1 - \nu) \varepsilon E - \gamma^{\text{sym}} I^{\text{sym}}",
        r"\frac{dI^{\text{sev}}}{dt} = (1 - \eta) \nu \varepsilon E - \left( (1 - \sigma(t)) \gamma^{\text{sev-r}} + \sigma(t) \gamma^{\text{sev-d}} \right) I^{\text{sev}}",
        r"\frac{dR}{dt} = \gamma^{\text{asym}} I^{\text{asym}} + \gamma^{\text{sym}} I^{\text{sym}} + (1 - \sigma(t)) \gamma^{\text{sev-r}} I^{\text{sev}}",
        r"\frac{dD}{dt} = \sigma(t) \gamma^{\text{sev-d}} I^{\text{sev}}"
    ]
    
    st.markdown(f"""
                # SEIIIRD Epidemiological Model

The SEIIIRD model is an extension of classical compartmental models used in epidemiology, incorporating various stages of infection and severity. It is particularly useful for modeling diseases where individuals can be asymptomatic, symptomatic, or severely infected, with possible outcomes of recovery or death.

## Compartments
- **S (Susceptible):** Individuals who are at risk of infection.
- **E (Exposed):** Individuals who have been exposed to the virus but are not yet infectious.
- **I_asym (Infected Asymptomatic):** Infected individuals who do not show symptoms but can still transmit the disease.
- **I_sym (Infected Symptomatic):** Infected individuals who develop symptoms.
- **I_sev (Infected Severe):** Infected individuals who experience severe symptoms and may require hospitalization.
- **R (Recovered):** Individuals who have recovered from the disease and are no longer infectious.
- **D (Dead):** Individuals who have died from the disease.

## Differential Equations
The following equations govern the dynamics of the system:

$$
{eqs[0]}
$$

$$
{eqs[1]}
$$

$$
{eqs[2]}
$$

$$
{eqs[3]}
$$

$$
{eqs[4]}
$$

$$
{eqs[5]}
$$

$$
{eqs[6]}
$$

## Explanation
- The **susceptible (S)** population decreases as individuals become exposed (E) through contact with infected individuals.
- The **exposed (E)** group transitions to either **asymptomatic infected (I_asym)** or **symptomatic infected (I_sym, I_sev)** at a rate $$ \epsilon $$.
- A fraction $$ \eta $$ of exposed individuals become **asymptomatic (I_asym)**, while the remaining become **symptomatic (I_sym) or severe (I_sev)** based on probability $$ \nu $$.
- Asymptomatic and symptomatic individuals recover at rates $$ \gamma^{{asym}} $$ and $$ \gamma^{{sym}} $$ respectively.
- Severely infected individuals either recover at a rate $$ \gamma^{{sev-r}} $$ or die at a rate $$ \gamma^{{sev-d}} $$, with $$ \sigma(t) $$ representing the probability of death.
- The **recovered (R)** population increases as individuals from **I_asym, I_sym, and I_sev** recover.
- The **dead (D)** population increases as severely infected individuals die.

This model provides a comprehensive framework for understanding disease progression and evaluating intervention strategies such as social distancing and vaccination.

                
                """, unsafe_allow_html=True)
    
    # Initialize session state
    if "simulation_type" not in st.session_state:
        st.session_state.simulation_type = None

    st.write("Choose simulation type:")

    # Create three columns: one for text, one for the first button, and one for the second button
    col1, col2, col3 = st.columns([1, 2, 2])

    with col2:
        if st.button("Variant Based"):
            st.session_state.simulation_type = "variant"

    with col3:
        if st.button("Parameter Based"):
            st.session_state.simulation_type = "parameter"
    
    if st.session_state.simulation_type == "parameter":
        with st.form("epidemic_form"):
            col1, col2 = st.columns(2)

            with col1:
                N = st.slider("No. of persons", 100, 1000, 500)
                beta = st.slider("Infectivity Rate", 0.0, 1.0, 0.3)

            with col2:
                sigma = st.slider("Death rate", 0.0, 1.0, 0.3)
                gamma_inv = st.slider("Infectious Period (Days)", 1, 20, 5)

            submitted = st.form_submit_button("Submit")

        if submitted:
            df = pd.read_csv("../simulation/data/simulation/1.csv", sep=";", header=None)
            df.loc[df[0] == 'N', 1] = N
            df.loc[df[0] == 'N_total', 1] = N
            df.loc[df[0] == 'beta_asym', 1] = beta
            df.loc[df[0] == 'beta_sym', 1] = beta
            df.loc[df[0] == 'beta_sev', 1] = beta
            df.loc[df[0] == 'sigma', 1] = sigma
            df.loc[df[0] == 'gamma_asym', 1] = 1.0 / gamma_inv
            df.loc[df[0] == 'gamma_sym', 1] = 1.0 / gamma_inv
            df.loc[df[0] == 'gamma_sev_d_hat', 1] = 1.0 / gamma_inv
            df.loc[df[0] == 'gamma_sev_r_hat', 1] = 1.0 / gamma_inv
            df.loc[df[0] == 'S', 1] = int(0.95 * N)
            df.loc[df[0] == 'E', 1] = int(0.02 * N)
            df.loc[df[0] == 'I_asym', 1] = int(0.01 * N)
            df.loc[df[0] == 'I_sym', 1] = int(0.01 * N)
            df.loc[df[0] == 'I_sev', 1] = int(0.01 * N)
            df.loc[df[0] == 'R', 1] = N - int(0.95 * N) - int(0.02 * N) - 3 * int(0.01 * N)
            df.loc[df[0] == 'D', 1] = 0.0
            df.loc[df[0] == 'beds', 1] = int(0.02 * N)
            
            df.to_csv("../simulation/data/simulation/1.csv", sep=";", header=False, index=False)
            
            result_dict, visualizer, animation_obj = run_simulation("simulation", "../simulation/data/simulation/", plot_results=False)
            
            fig = visualizer.plot_aggregated_curves(return_fig=True)
            
            st.plotly_chart(fig)
            
            deaths, icu, duration = read_result_file("../results/simulation/1.res")
            st.write(f"Total deaths: {deaths}")
            st.write(f"Days with exceeded ICU bed capacity: {icu}")
            st.write(f"Duration of pandemic (in days): {duration}")
            
            
            animation_obj.save_gif('animation.gif')
            if 'animation_displayed' not in st.session_state:
                col1, col2 = st.columns([5, 3])  # Adjust column widths as needed
                with col1:
                    st.image('animation.gif')
                with col2:
                    st.markdown("""
                                **S**: blue  
                                **E**: orange  
                                **I_asym**: red  
                                **I_sym**: dark red  
                                **I_sev**: purple  
                                **R**: green  
                                **D**: black  
                                """)
                st.session_state['animation_displayed'] = True
            
    if st.session_state.simulation_type == "variant":
        variant = st.selectbox("Select Variant", ["Beta", "Gamma", "Alpha", "Delta", "Omicron"])
        st.write("Death Rate Order: Beta > Gamma > Alpha > Delta > Omicron")
        
        result_dict, visualizer, animation_obj = run_simulation(variant, "../simulation/data/"+variant+"/", plot_results=False)
        fig = visualizer.plot_aggregated_curves(return_fig=True)
        
        st.plotly_chart(fig)
        
        deaths, icu, duration = read_result_file(f"../results/{variant}/a.res")
        st.write(f"Total deaths: {deaths}")
        st.write(f"Days with exceeded ICU bed capacity: {icu}")
        st.write(f"Duration of pandemic (in days): {duration}")