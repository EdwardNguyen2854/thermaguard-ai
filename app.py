import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import joblib
import time
from pathlib import Path
from datetime import datetime
import io

from src.data.load import load_hvac_data, load_csv, load_parquet, save_parquet
from src.data.clean import clean_hvac_data
from src.features.build_features import engineer_features
from src.models.predictive_maintenance import (
    create_failure_labels,
    balance_classes,
    prepare_train_test_split,
    train_random_forest,
    train_xgboost_model,
    train_lightgbm_model,
    evaluate_classifier,
    get_feature_importance,
    save_model as save_ml_model,
    load_model as load_ml_model,
    train_ensemble,
    ensemble_predict
)
from src.models.anomaly import (
    zscore_anomaly_detection,
    train_isolation_forest,
    isolation_forest_predict
)
from src.models.energy import (
    profile_energy_consumption,
    calculate_efficiency_metrics,
    recommend_setpoints,
    optimize_schedule
)

st.set_page_config(
    page_title="ThermaGuard AI - Training Platform",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
MODELS_DIR.mkdir(exist_ok=True)

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None
if "training_history" not in st.session_state:
    st.session_state.training_history = []


def render_metrics_card(metrics: dict, col1, col2):
    """Render evaluation metrics in columns."""
    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    with col2:
        st.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.2%}")


def plot_confusion_matrix(cm, labels):
    """Plot confusion matrix using plotly."""
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True
    )
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        margin=dict(l=150, r=50, t=50, b=150)
    )
    return fig


def plot_feature_importance(importance_df):
    """Plot feature importance as horizontal bar chart."""
    fig = px.bar(
        importance_df.head(15)[::-1],
        x='importance',
        y='feature',
        orientation='h',
        title="Top 15 Feature Importances"
    )
    fig.update_layout(
        yaxis_title="Feature",
        xaxis_title="Importance",
        height=500
    )
    return fig


def plot_training_history():
    """Plot training history across all sessions."""
    if not st.session_state.training_history:
        return None
    
    df = pd.DataFrame(st.session_state.training_history)
    fig = px.line(
        df,
        x="timestamp",
        y="accuracy",
        color="model_name",
        markers=True,
        title="Training Accuracy Over Time"
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Accuracy",
        height=400
    )
    return fig


st.title("🏭 ThermaGuard AI - Model Training Platform")
st.markdown("### Train and deploy machine learning models for HVAC predictive maintenance")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Management",
    "🧠 Model Training",
    "🔮 Predictions",
    "📈 Analytics"
])

with tab1:
    st.header("Data Management")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Data")
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload HVAC sensor data in CSV format"
        )
        
        if uploaded_file:
            try:
                df_uploaded = load_csv(io.BytesIO(uploaded_file.getvalue()))
                st.success(f"Loaded {len(df_uploaded)} rows")
                st.dataframe(df_uploaded.head())
            except Exception as e:
                st.error(f"Error loading file: {e}")
        
        st.divider()
        
        st.subheader("Use Sample Data")
        if st.button("Load Turin HVAC Sample Data"):
            try:
                df = load_hvac_data(use_cached=True)
                st.success(f"Loaded cached data: {len(df)} rows")
                st.session_state.data = df
            except Exception as e:
                st.error(f"Error loading data: {e}")
    
    with col2:
        st.subheader("Data Preview")
        
        if 'data' not in st.session_state:
            st.info("Upload a CSV file or load sample data to begin")
        else:
            df = st.session_state.data
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Records", f"{len(df):,}")
            with col_b:
                st.metric("Date Range", f"{df['Timestamp'].dt.date.min()} to {df['Timestamp'].dt.date.max()}")
            with col_c:
                st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            
            st.divider()
            
            show_cols = st.multiselect(
                "Columns to display",
                options=df.columns.tolist(),
                default=['Timestamp', 'T_Supply', 'T_Return', 'Power']
            )
            
            n_rows = st.slider("Number of rows", 10, 500, 100)
            st.dataframe(df[show_cols].head(n_rows), height=300)
            
            st.subheader("Statistics")
            st.dataframe(df[show_cols].describe(), height=200)


with tab2:
    st.header("Model Training")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Training Configuration")
        
        model_type = st.selectbox(
            "Select Model Type",
            ["Random Forest", "XGBoost", "LightGBM", "Ensemble (RF+XGB+LGB)"],
            help="Choose the model algorithm to train"
        )
        
        st.divider()
        
        if model_type == "Random Forest":
            n_estimators = st.slider("Number of Trees", 50, 500, 100, 10)
            max_depth = st.slider("Max Depth", 5, 30, 15)
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)
            
            model_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf
            }
        elif model_type == "XGBoost":
            n_estimators = st.slider("Number of Trees", 50, 500, 100, 10)
            max_depth = st.slider("Max Depth", 3, 15, 6)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
            
            model_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate
            }
        elif model_type == "LightGBM":
            n_estimators = st.slider("Number of Trees", 50, 500, 100, 10)
            max_depth = st.slider("Max Depth", 3, 15, 6)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
            
            model_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate
            }
        else:
            model_params = {}
        
        st.divider()
        
        st.subheader("Data Settings")
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
        balance_method = st.selectbox(
            "Class Balancing",
            ["None", "Undersample"],
            help="Handle imbalanced classes"
        )
        failure_threshold = st.number_input(
            "Failure Threshold (kW)",
            value=0.0,
            help="Power threshold for failure detection"
        )
        window_size = st.slider(
            "Failure Window Size",
            4, 96, 24,
            help="Rolling window for failure detection"
        )
        
        st.divider()
        
        col_a, col_b = st.columns(2)
        with col_a:
            train_button = st.button("🧠 Train Model", type="primary", use_container_width=True)
        with col_b:
            model_name = st.text_input("Model Name", value=f"{model_type.lower().replace(' ', '_')}_{datetime.now().strftime('%H%M%S')}")
    
    with col2:
        if train_button:
            if 'data' not in st.session_state:
                st.error("Please load data first!")
            else:
                with st.spinner("Training model..."):
                    df = st.session_state.data.copy()
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Creating failure labels...")
                    progress_bar.progress(20)
                    df = create_failure_labels(
                        df,
                        target_col="Power",
                        threshold=failure_threshold,
                        window_size=window_size
                    )
                    
                    status_text.text("Preparing train/test split...")
                    progress_bar.progress(40)
                    
                    feature_cols = [c for c in df.columns 
                                   if c not in ['failure_imminent', 'rul', 'is_failure', 'failure_window', 'Timestamp'] 
                                   and pd.api.types.is_numeric_dtype(df[c])]
                    st.session_state.feature_cols = feature_cols
                    
                    X_train, X_test, y_train, y_test = prepare_train_test_split(
                        df,
                        target_col="failure_imminent",
                        test_size=test_size,
                        feature_cols=feature_cols
                    )
                    
                    status_text.text("Balancing classes...")
                    progress_bar.progress(50)
                    
                    if balance_method == "Undersample":
                        X_train, y_train = balance_classes(X_train, y_train, "undersample")
                    
                    status_text.text("Training model...")
                    progress_bar.progress(60)
                    
                    start_time = time.time()
                    
                    if model_type == "Random Forest":
                        model = train_random_forest(X_train, y_train, **model_params)
                    elif model_type == "XGBoost":
                        model = train_xgboost_model(X_train, y_train, **model_params)
                    elif model_type == "LightGBM":
                        model = train_lightgbm_model(X_train, y_train, **model_params)
                    else:
                        models_dict = train_ensemble(X_train, y_train)
                        model = {"ensemble": models_dict}
                    
                    training_time = time.time() - start_time
                    
                    status_text.text("Evaluating model...")
                    progress_bar.progress(80)
                    
                    if model_type == "Ensemble (RF+XGB+LGB)":
                        X_test_clean = X_test.fillna(0).replace([np.inf, -np.inf], 0)
                        y_pred_proba = ensemble_predict(models_dict, X_test_clean)
                        y_pred = (y_pred_proba > 0.5).astype(int)
                        metrics = {
                            "accuracy": float((y_pred == y_test).mean()),
                            "roc_auc": float(pd.Series(y_pred_proba).sum())
                        }
                    else:
                        X_test_clean = X_test.fillna(0).replace([np.inf, -np.inf], 0)
                        metrics = evaluate_classifier(model, X_test_clean, y_test)
                    
                    status_text.text("Saving model...")
                    progress_bar.progress(90)
                    
                    model_path = MODELS_DIR / f"{model_name}.joblib"
                    if model_type == "Ensemble (RF+XGB+LGB)":
                        joblib.dump(models_dict, model_path)
                    else:
                        save_ml_model(model, str(model_path))
                    
                    st.session_state.trained_models[model_name] = {
                        "model": model,
                        "model_type": model_type,
                        "metrics": metrics,
                        "feature_cols": feature_cols,
                        "path": str(model_path),
                        "timestamp": datetime.now()
                    }
                    st.session_state.current_model = model_name
                    
                    progress_bar.progress(100)
                    status_text.text("Training complete!")
                    
                    st.session_state.training_history.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "model_name": model_name,
                        "model_type": model_type,
                        "accuracy": metrics.get("accuracy", 0),
                        "roc_auc": metrics.get("roc_auc", 0),
                        "training_time": training_time
                    })
                    
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"Model trained successfully in {training_time:.2f}s!")
        
        st.subheader("Training Results")
        
        if st.session_state.current_model and st.session_state.current_model in st.session_state.trained_models:
            model_info = st.session_state.trained_models[st.session_state.current_model]
            metrics = model_info["metrics"]
            
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            with col_b:
                st.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.2f}" if metrics.get('roc_auc') else "N/A")
            with col_c:
                st.metric("Model Type", model_info["model_type"])
            with col_d:
                st.metric("Features", len(model_info["feature_cols"]))
            
            st.divider()
            
            col_a, col_b = st.columns(2)
            with col_a:
                if 'confusion_matrix' in metrics:
                    cm = metrics['confusion_matrix']
                    fig = plot_confusion_matrix(cm, ['Normal', 'Failure'])
                    st.plotly_chart(fig, use_container_width=True)
            
            with col_b:
                if model_info["model_type"] != "Ensemble (RF+XGB+LGB)":
                    model = model_info["model"]
                    importance_df = get_feature_importance(model, model_info["feature_cols"])
                    if not importance_df.empty:
                        fig = plot_feature_importance(importance_df)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train a model to see results here")
        
        st.divider()
        
        st.subheader("Training History")
        if st.session_state.training_history:
            fig = plot_training_history()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            hist_df = pd.DataFrame(st.session_state.training_history)
            st.dataframe(hist_df, height=200)
        else:
            st.info("No training history yet")


with tab3:
    st.header("Predictions")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Selection")
        
        available_models = list(st.session_state.trained_models.keys())
        if not available_models:
            st.warning("No trained models available. Train a model first!")
            model_to_use = None
        else:
            selected = st.selectbox(
                "Select Model",
                ["Latest"] + available_models,
                help="Choose a trained model for predictions"
            )
            
            if selected == "Latest":
                model_to_use = st.session_state.current_model
            else:
                model_to_use = selected
            
            if model_to_use and model_to_use in st.session_state.trained_models:
                model_info = st.session_state.trained_models[model_to_use]
                st.caption(f"Type: {model_info['model_type']}")
                st.caption(f"Accuracy: {model_info['metrics'].get('accuracy', 0):.2%}")
        
        st.divider()
        
        st.subheader("Input Features")
        
        if 'data' in st.session_state:
            df = st.session_state.data
            default_vals = {
                'T_Supply': df['T_Supply'].median(),
                'T_Return': df['T_Return'].median(),
                'SP_Return': df['SP_Return'].median(),
                'T_Saturation': df['T_Saturation'].median(),
                'T_Outdoor': df['T_Outdoor'].median(),
                'RH_Supply': df['RH_Supply'].median(),
                'RH_Return': df['RH_Return'].median(),
                'RH_Outdoor': df['RH_Outdoor'].median(),
                'Energy': df['Energy'].median(),
                'Power': df['Power'].median()
            }
        else:
            default_vals = {
                'T_Supply': 20.0, 'T_Return': 21.0, 'SP_Return': 20.0,
                'T_Saturation': 19.0, 'T_Outdoor': 15.0,
                'RH_Supply': 70.0, 'RH_Return': 55.0, 'RH_Outdoor': 80.0,
                'Energy': 100.0, 'Power': 2.5
            }
        
        input_data = {}
        
        input_data['T_Supply'] = st.slider("Supply Temperature (°C)", -10.0, 50.0, default_vals['T_Supply'])
        input_data['T_Return'] = st.slider("Return Temperature (°C)", -10.0, 50.0, default_vals['T_Return'])
        input_data['SP_Return'] = st.slider("Setpoint Return (°C)", -10.0, 50.0, default_vals['SP_Return'])
        input_data['T_Saturation'] = st.slider("Saturation Temperature (°C)", -10.0, 50.0, default_vals['T_Saturation'])
        input_data['T_Outdoor'] = st.slider("Outdoor Temperature (°C)", -30.0, 60.0, default_vals['T_Outdoor'])
        input_data['RH_Supply'] = st.slider("Supply Humidity (%)", 0.0, 100.0, default_vals['RH_Supply'])
        input_data['RH_Return'] = st.slider("Return Humidity (%)", 0.0, 100.0, default_vals['RH_Return'])
        input_data['RH_Outdoor'] = st.slider("Outdoor Humidity (%)", 0.0, 100.0, default_vals['RH_Outdoor'])
        input_data['Energy'] = st.number_input("Energy (kWh)", min_value=0.0, value=default_vals['Energy'])
        input_data['Power'] = st.number_input("Power (kW)", min_value=0.0, value=default_vals['Power'])
        
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            if not model_to_use:
                st.error("Please select a trained model")
            else:
                model_info = st.session_state.trained_models[model_to_use]
                model = model_info["model"]
                feature_cols = model_info["feature_cols"]
                
                X = pd.DataFrame([input_data])
                
                for col in feature_cols:
                    if col not in X.columns:
                        X[col] = 0
                
                X = X[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
                
                if model_info["model_type"] == "Ensemble (RF+XGB+LGB)":
                    proba = ensemble_predict(model, X)[0]
                    prediction = 1 if proba > 0.5 else 0
                else:
                    prediction = model.predict(X)[0]
                    proba = model.predict_proba(X)[0][1]
                
                st.session_state.last_prediction = {
                    "prediction": prediction,
                    "probability": proba,
                    "timestamp": datetime.now()
                }
        
        st.divider()
        
        st.subheader("Batch Predictions")
        
        if st.button("Predict on Recent Data", use_container_width=True):
            if 'data' not in st.session_state or not model_to_use:
                st.error("Please load data and select a model first")
            else:
                df = st.session_state.data.tail(500).copy()
                model_info = st.session_state.trained_models[model_to_use]
                feature_cols = model_info["feature_cols"]
                
                X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
                
                if model_info["model_type"] == "Ensemble (RF+XGB+LGB)":
                    probas = ensemble_predict(model, X)
                    predictions = (probas > 0.5).astype(int)
                else:
                    predictions = model.predict(X)
                    probas = model.predict_proba(X)[:, 1]
                
                st.session_state.batch_predictions = pd.DataFrame({
                    'Timestamp': df['Timestamp'].values,
                    'Prediction': ['Failure' if p == 1 else 'Normal' for p in predictions],
                    'Probability': probas
                })
                
                st.success(f"Generated {len(predictions)} predictions")
    
    with col2:
        st.subheader("Prediction Result")
        
        if 'last_prediction' in st.session_state:
            pred = st.session_state.last_prediction
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                if pred['prediction'] == 1:
                    st.error("⚠️ **FAILURE PREDICTED**")
                else:
                    st.success("✅ **NORMAL OPERATION**")
            
            with col_b:
                st.metric("Failure Probability", f"{pred['probability']:.2%}")
            
            st.divider()
            
            prob = pred['probability']
            fig = go.Figure()
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': 'lightgreen'},
                        {'range': [30, 70], 'color': 'yellow'},
                        {'range': [70, 100], 'color': 'red'}
                    ]
                },
                title={'text': "Risk Level"}
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            if pred['prediction'] == 1:
                st.warning("""
                **Recommendations:**
                1. Schedule immediate maintenance inspection
                2. Check compressor operation and refrigerant levels
                3. Verify heat exchange performance
                4. Inspect electrical connections
                """)
        else:
            st.info("Make a prediction using the input controls")
        
        st.divider()
        
        st.subheader("Batch Prediction Results")
        
        if 'batch_predictions' in st.session_state:
            df_preds = st.session_state.batch_predictions
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                failure_count = (df_preds['Prediction'] == 'Failure').sum()
                st.metric("Failures Detected", failure_count)
            with col_b:
                st.metric("Normal Operations", len(df_preds) - failure_count)
            with col_c:
                st.metric("Avg Probability", f"{df_preds['Probability'].mean():.2%}")
            
            st.dataframe(df_preds, height=300)
            
            fig = px.scatter(
                df_preds.head(100),
                x=range(len(df_preds.head(100))),
                y='Probability',
                color='Prediction',
                color_discrete_map={'Failure': 'red', 'Normal': 'green'},
                title="Prediction Probabilities (Last 100)"
            )
            fig.update_layout(
                xaxis_title="Sample",
                yaxis_title="Failure Probability"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run batch predictions to see results")


with tab4:
    st.header("Analytics Dashboard")
    
    if 'data' not in st.session_state:
        st.warning("Please load data first")
    else:
        df = st.session_state.data
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            date_range = f"{df['Timestamp'].dt.date.min()} to {df['Timestamp'].dt.date.max()}"
            st.metric("Date Range", date_range[:20] + "...")
        with col3:
            duty_cycle = (df['Power'] > 0).mean() * 100
            st.metric("Duty Cycle", f"{duty_cycle:.1f}%")
        with col4:
            st.metric("Avg Power", f"{df['Power'].mean():.2f} kW")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Power Consumption Over Time")
            sample_df = df.iloc[::100].copy()
            fig = px.line(
                sample_df,
                x='Timestamp',
                y='Power',
                title='Power Consumption Trend'
            )
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Power (kW)",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Temperature Distribution")
            fig = px.histogram(
                df,
                x='T_Supply',
                nbins=50,
                title='Supply Temperature Distribution'
            )
            fig.update_layout(
                xaxis_title="Temperature (°C)",
                yaxis_title="Count",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Daily Power Pattern")
            df['hour'] = pd.to_datetime(df['Timestamp']).dt.hour
            hourly_avg = df.groupby('hour')['Power'].mean().reset_index()
            
            fig = px.bar(
                hourly_avg,
                x='hour',
                y='Power',
                labels={'hour': 'Hour of Day', 'Power': 'Avg Power (kW)'}
            )
            fig.update_layout(
                xaxis=dict(tickmode='linear', tick0=0, dtick=1),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Temperature Relationship")
            fig = px.scatter(
                df.sample(min(1000, len(df))),
                x='T_Outdoor',
                y='Power',
                color='T_Supply',
                title='Power vs Outdoor Temperature'
            )
            fig.update_layout(
                xaxis_title="Outdoor Temperature (°C)",
                yaxis_title="Power (kW)",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Weekly Pattern")
            df['day_of_week'] = pd.to_datetime(df['Timestamp']).dt.dayofweek
            daily = df.groupby('day_of_week')['Power'].mean().reset_index()
            daily['day_name'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            fig = px.bar(
                daily,
                x='day_name',
                y='Power',
                labels={'day_name': 'Day', 'Power': 'Avg Power (kW)'},
                color='Power',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Anomaly Overview")
            df['is_anomaly'] = df['Power'] > df['Power'].quantile(0.95)
            anomaly_count = df['is_anomaly'].sum()
            normal_count = len(df) - anomaly_count
            
            fig = px.pie(
                values=[normal_count, anomaly_count],
                names=['Normal', 'Anomaly'],
                title='Anomaly Distribution',
                hole=0.4
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        st.subheader("Energy Optimization Insights")
        
        peak_hours = [6, 7, 8, 9, 10, 11]
        off_peak = [0, 1, 2, 21, 22, 23]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Peak Hours:** {peak_hours}
            
            Highest consumption periods. Consider:
            - Pre-cooling during off-peak hours
            - Shifting non-critical loads
            - Using thermal storage
            """)
        
        with col2:
            st.success(f"""
            **Optimal Off-Peak Hours:** {off_peak}
            
            Best times for:
            - Heavy equipment operation
            - Scheduled maintenance
            - Thermal storage charging
            """)
        
        avg_power_off_peak = df[df['hour'].isin(off_peak)]['Power'].mean()
        avg_power_peak = df[df['hour'].isin(peak_hours)]['Power'].mean()
        savings_potential = (avg_power_peak - avg_power_off_peak) * len(off_peak) * 365 * 0.12
        
        st.metric("Estimated Annual Savings Potential", f"${savings_potential:.2f}")

st.sidebar.header("Settings")

with st.sidebar:
    st.subheader("Saved Models")
    
    for name, info in st.session_state.trained_models.items():
        with st.expander(name):
            st.caption(f"Type: {info['model_type']}")
            st.caption(f"Accuracy: {info['metrics'].get('accuracy', 0):.2%}")
            st.caption(f"Trained: {info['timestamp'].strftime('%H:%M:%S')}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load", key=f"load_{name}"):
                    st.session_state.current_model = name
                    st.success(f"Loaded {name}")
            with col2:
                if st.button("Delete", key=f"del_{name}"):
                    del st.session_state.trained_models[name]
                    if st.session_state.current_model == name:
                        st.session_state.current_model = None
                    st.success(f"Deleted {name}")
    
    st.divider()
    
    if st.button("Clear All Models"):
        st.session_state.trained_models = {}
        st.session_state.current_model = None
        st.success("All models cleared")
    
    if st.button("Export Models"):
        if st.session_state.trained_models:
            st.info("Models are saved in the /models directory")
        else:
            st.warning("No models to export")

st.caption("ThermaGuard AI Training Platform v1.0")