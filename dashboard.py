import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

st.set_page_config(page_title="ThermaGuard AI Dashboard", layout="wide")

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

@st.cache_data
def load_data():
    """Load processed data."""
    df = pd.read_parquet(DATA_DIR / "processed" / "turin_features.parquet")
    return df

@st.cache_data
def load_clean_data():
    """Load cleaned data."""
    return pd.read_parquet(DATA_DIR / "interim" / "turin_clean.parquet")

@st.cache_resource
def load_model():
    """Load trained model."""
    return joblib.load(MODELS_DIR / "random_forest.joblib")

st.title("ThermaGuard AI Dashboard")
st.markdown("### Predictive Maintenance & Energy Optimization for HVAC Systems")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Data Explorer", "Predictions", "Energy Analysis"])

with tab1:
    st.header("System Overview")
    
    df_clean = load_clean_data()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df_clean):,}")
    with col2:
        st.metric("Date Range", f"{df_clean['Timestamp'].dt.date.min()} to {df_clean['Timestamp'].dt.date.max()}")
    with col3:
        duty_cycle = (df_clean['Power'] > 0).mean() * 100
        st.metric("Duty Cycle", f"{duty_cycle:.1f}%")
    with col4:
        st.metric("Avg Power", f"{df_clean['Power'].mean():.2f} kW")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Power Consumption Over Time")
        df_sample = df_clean.iloc[::100]
        fig = px.line(df_sample, x='Timestamp', y='Power', title='Power Consumption')
        fig.update_layout(xaxis_title="Time", yaxis_title="Power (kW)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Temperature Distribution")
        fig = px.histogram(df_clean, x='T_Supply', nbins=50, title='Supply Temperature')
        fig.update_layout(xaxis_title="Temperature (°C)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Daily Power Pattern")
        df_clean['hour'] = pd.to_datetime(df_clean['Timestamp']).dt.hour
        hourly_avg = df_clean.groupby('hour')['Power'].mean()
        fig = px.bar(x=hourly_avg.index, y=hourly_avg.values, 
                     labels={'x': 'Hour of Day', 'y': 'Average Power (kW)'})
        fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Anomaly Detection Results")
        df_anomaly = df_clean.copy()
        df_anomaly['is_anomaly'] = df_anomaly['Power'] > df_anomaly['Power'].quantile(0.95)
        anomaly_count = df_anomaly['is_anomaly'].sum()
        normal_count = len(df_anomaly) - anomaly_count
        
        fig = px.pie(values=[normal_count, anomaly_count], 
                     names=['Normal', 'Anomaly'],
                     title='Anomaly Distribution')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Data Explorer")
    
    df = load_data()
    
    st.subheader("Filter Data")
    col1, col2 = st.columns(2)
    
    with col1:
        date_range = st.slider(
            "Select Date Range",
            min_value=0,
            max_value=len(df)-1,
            value=(0, min(1000, len(df)-1))
        )
    
    with col2:
        columns_to_show = st.multiselect(
            "Select Columns",
            options=df.columns.tolist()[:20],
            default=['Timestamp', 'T_Supply', 'T_Return', 'Power']
        )
    
    st.dataframe(df.iloc[date_range[0]:date_range[1]][columns_to_show], height=300)
    
    st.subheader("Statistics")
    st.dataframe(df_clean[columns_to_show].describe())

with tab3:
    st.header("Failure Predictions")
    
    model = load_model()
    df = load_data()
    
    st.subheader("Make a Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        t_supply = st.slider("Supply Temperature (°C)", -10.0, 50.0, 20.0)
        t_return = st.slider("Return Temperature (°C)", -10.0, 50.0, 21.0)
        sp_return = st.slider("Setpoint Return (°C)", -10.0, 50.0, 20.0)
        t_saturation = st.slider("Saturation Temperature (°C)", -10.0, 50.0, 19.0)
        t_outdoor = st.slider("Outdoor Temperature (°C)", -30.0, 60.0, 15.0)
    
    with col2:
        rh_supply = st.slider("Supply Humidity (%)", 0.0, 100.0, 70.0)
        rh_return = st.slider("Return Humidity (%)", 0.0, 100.0, 55.0)
        rh_outdoor = st.slider("Outdoor Humidity (%)", 0.0, 100.0, 80.0)
        energy = st.number_input("Energy (kWh)", min_value=0.0, value=100.0)
        power = st.number_input("Power (kW)", min_value=0.0, value=2.5)
    
    if st.button("Predict Failure"):
        features = {
            'T_Supply': t_supply, 'T_Return': t_return, 'SP_Return': sp_return,
            'T_Saturation': t_saturation, 'T_Outdoor': t_outdoor,
            'RH_Supply': rh_supply, 'RH_Return': rh_return, 'RH_Outdoor': rh_outdoor,
            'Energy': energy, 'Power': power
        }
        
        X = pd.DataFrame([features])
        
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error(f"⚠️ FAILURE PREDICTED")
            else:
                st.success(f"✅ NORMAL OPERATION")
        
        with col2:
            st.metric("Failure Probability", f"{proba:.2%}")
        
        if prediction == 1:
            st.warning("""
            **Recommendations:**
            - Schedule immediate maintenance inspection
            - Check compressor operation
            - Verify heat exchange performance
            """)
    
    st.divider()
    
    st.subheader("Recent Predictions")
    df = load_data()
    recent_data = df.tail(100).copy()
    
    exclude = ['failure_imminent', 'rul', 'is_failure', 'failure_window', 'Timestamp']
    feature_cols = [c for c in recent_data.columns if c not in exclude and pd.api.types.is_numeric_dtype(recent_data[c])]
    
    X = recent_data[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    results = recent_data[['Timestamp']].copy()
    results['Prediction'] = ['Failure' if p == 1 else 'Normal' for p in predictions]
    results['Probability'] = probabilities
    
    st.dataframe(results, height=300)

with tab4:
    st.header("Energy Analysis")
    
    df = load_clean_data()
    
    st.subheader("Energy Consumption Profile")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Energy", f"{df['Energy'].max():.0f} kWh")
    with col2:
        st.metric("Avg Power", f"{df['Power'].mean():.2f} kW")
    with col3:
        st.metric("Peak Power", f"{df['Power'].max():.2f} kW")
    with col4:
        on_hours = (df['Power'] > 0).sum()
        st.metric("Operating Hours", f"{on_hours:,}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hourly Consumption Pattern")
        df['hour'] = pd.to_datetime(df['Timestamp']).dt.hour
        hourly = df.groupby('hour')['Power'].agg(['mean', 'sum']).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=hourly['hour'], y=hourly['mean'], name='Avg Power'))
        fig.update_layout(xaxis_title="Hour", yaxis_title="Power (kW)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Weekly Pattern")
        df['day_of_week'] = pd.to_datetime(df['Timestamp']).dt.dayofweek
        daily = df.groupby('day_of_week')['Power'].mean().reset_index()
        daily['day_name'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig = px.bar(daily, x='day_name', y='Power', 
                     labels={'day_name': 'Day', 'Power': 'Avg Power (kW)'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("Optimization Recommendations")
    
    peak_hours = [6, 7, 8, 9, 10, 11]
    off_peak = [0, 1, 2, 21, 22, 23]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Peak Hours to Avoid:** {peak_hours}
        
        These hours show highest consumption. Consider:
        - Pre-cooling/pre-heating during off-peak
        - Shifting non-critical loads
        """)
    
    with col2:
        st.success(f"""
        **Optimal Off-Peak Hours:** {off_peak}
        
        Recommended for:
        - Running heavy equipment
        - Scheduling maintenance
        - Charging thermal storage
        """)
    
    st.divider()
    
    savings = (df['Power'].mean() * 0.15) * 24 * 365 * 0.12
    st.metric("Estimated Annual Savings Potential", f"${savings:.2f}")

st.divider()
st.caption("ThermaGuard AI - Industrial ML Platform for HVAC Systems")
