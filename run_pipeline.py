import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.load import load_hvac_data, save_parquet, get_interim_data_path, get_processed_data_path
from src.data.clean import clean_hvac_data, get_data_quality_report
from src.features.build_features import engineer_features
from src.models.predictive_maintenance import (
    create_failure_labels, prepare_train_test_split, train_random_forest,
    train_xgboost_model, evaluate_classifier, save_model
)
from src.models.anomaly import (
    zscore_anomaly_detection, train_isolation_forest, isolation_forest_predict,
    ensemble_anomaly_score, classify_alert_severity
)
from src.models.energy import generate_optimization_report
import pandas as pd
import numpy as np


def run_phase1_data_infrastructure():
    """Run Phase 1: Data Infrastructure."""
    print("=" * 60)
    print("Phase 1: Data Infrastructure & Exploration")
    print("=" * 60)
    
    print("\n[1.1] Loading raw data...")
    df = load_hvac_data(use_cached=False)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\n[1.2] Data quality report (raw)...")
    quality_raw = get_data_quality_report(df)
    print(f"Total rows: {quality_raw['total_rows']}")
    print(f"Missing values: {sum(v['count'] for v in quality_raw['missing_values'].values())}")
    
    print("\n[1.3] Cleaning data...")
    df_clean = clean_hvac_data(df, fill_method="forward", remove_outliers=True, ensure_continuity=False)
    print(f"After cleaning: {len(df_clean)} rows")
    
    print("\n[1.4] Saving cleaned data...")
    save_parquet(df_clean, get_interim_data_path())
    print(f"Saved to {get_interim_data_path()}")
    
    quality_clean = get_data_quality_report(df_clean)
    print(f"Missing values after cleaning: {sum(v['count'] for v in quality_clean['missing_values'].values())}")
    
    return df_clean


def run_phase2_feature_engineering(df):
    """Run Phase 2: Feature Engineering."""
    print("\n" + "=" * 60)
    print("Phase 2: Feature Engineering")
    print("=" * 60)
    
    print("\n[2.1] Engineering features...")
    df_features = engineer_features(df, timestamp_col="Timestamp")
    print(f"After feature engineering: {len(df_features.columns)} columns")
    
    print("\n[2.2] Saving feature-engineered data...")
    save_parquet(df_features, get_processed_data_path())
    print(f"Saved to {get_processed_data_path()}")
    
    return df_features


def run_phase3_predictive_maintenance(df):
    """Run Phase 3: Predictive Maintenance Models."""
    print("\n" + "=" * 60)
    print("Phase 3: Predictive Maintenance Models")
    print("=" * 60)
    
    print("\n[3.1] Creating failure labels...")
    df_labeled = create_failure_labels(df, target_col="Power", threshold=0.0, window_size=24)
    failure_rate = df_labeled['failure_imminent'].mean()
    print(f"Failure rate: {failure_rate:.2%}")
    
    print("\n[3.2] Preparing train/test split...")
    df_labeled = df_labeled.ffill().fillna(0)
    df_labeled = df_labeled.replace([np.inf, -np.inf], np.nan).fillna(0)
    exclude = ['failure_imminent', 'rul', 'is_failure', 'failure_window', 'Timestamp']
    feature_cols = [c for c in df_labeled.columns if c not in exclude and pd.api.types.is_numeric_dtype(df_labeled[c])]
    print(f"Using {len(feature_cols)} features")
    
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        df_labeled, target_col='failure_imminent', test_size=0.2, feature_cols=feature_cols
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    print("\n[3.3] Training Random Forest model...")
    rf_model = train_random_forest(X_train, y_train, n_estimators=50)
    rf_metrics = evaluate_classifier(rf_model, X_test, y_test)
    print(f"RF Accuracy: {rf_metrics['accuracy']:.4f}")
    
    print("\n[3.4] Training XGBoost model...")
    xgb_model = train_xgboost_model(X_train, y_train)
    xgb_metrics = evaluate_classifier(xgb_model, X_test, y_test)
    print(f"XGB Accuracy: {xgb_metrics['accuracy']:.4f}")
    
    print("\n[3.5] Saving models...")
    save_model(rf_model, "models/random_forest.joblib")
    save_model(xgb_model, "models/xgboost_model.joblib")
    print("Models saved to models/")
    
    return df_labeled, rf_model


def run_phase4_anomaly_detection(df):
    """Run Phase 4: Anomaly Detection."""
    print("\n" + "=" * 60)
    print("Phase 4: Anomaly Detection")
    print("=" * 60)
    
    numeric_cols = ['T_Supply', 'T_Return', 'T_Saturation', 'T_Outdoor', 'Power']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    print("\n[4.1] Z-score anomaly detection...")
    df_anomalies = zscore_anomaly_detection(df, numeric_cols, threshold=3.0)
    zscore_anomalies = df_anomalies['zscore_anomaly_count'].sum()
    print(f"Z-score anomalies detected: {zscore_anomalies}")
    
    print("\n[4.2] Training Isolation Forest...")
    if_model, scaler = train_isolation_forest(df, numeric_cols, contamination=0.1)
    df_anomalies = isolation_forest_predict(if_model, scaler, df_anomalies, numeric_cols)
    if_anomalies = df_anomalies['if_anomaly'].sum()
    print(f"Isolation Forest anomalies: {if_anomalies}")
    
    print("\n[4.3] Ensemble scoring...")
    df_anomalies = ensemble_anomaly_score(df_anomalies)
    df_anomalies = classify_alert_severity(df_anomalies)
    
    severity_counts = df_anomalies['severity'].value_counts()
    print(f"Severity distribution: {severity_counts.to_dict()}")
    
    return df_anomalies


def run_phase5_energy_optimization(df):
    """Run Phase 5: Energy Optimization."""
    print("\n" + "=" * 60)
    print("Phase 5: Energy Optimization")
    print("=" * 60)
    
    print("\n[5.1] Generating energy optimization report...")
    report = generate_optimization_report(df, power_col="Power")
    
    print("\nConsumption Profile:")
    print(f"  Duty cycle: {report['consumption_profile']['duty_cycle_percent']:.2f}%")
    print(f"  Average power: {report['consumption_profile']['average_power']:.4f} kW")
    print(f"  Peak power: {report['consumption_profile']['peak_power']:.4f} kW")
    
    print("\nSchedule Optimization:")
    print(f"  Peak hours to avoid: {report['schedule_optimization']['peak_hours_to_avoid']}")
    print(f"  Optimal off-peak hours: {report['schedule_optimization']['optimal_off_peak_hours']}")
    
    print("\nEstimated Savings:")
    savings = report['estimated_savings']
    print(f"  Daily savings: {savings['daily_savings_kwh']:.2f} kWh")
    print(f"  Annual savings: {savings['annual_savings_kwh']:.2f} kWh")
    print(f"  Annual cost savings: ${savings['annual_savings_cost']:.2f}")
    
    return report


def main():
    """Run the complete ThermaGuard AI pipeline."""
    print("\n" + "=" * 60)
    print("ThermaGuard AI - Complete Pipeline")
    print("=" * 60)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    df_clean = run_phase1_data_infrastructure()
    
    df_features = run_phase2_feature_engineering(df_clean)
    
    df_labeled, rf_model = run_phase3_predictive_maintenance(df_features)
    
    df_anomalies = run_phase4_anomaly_detection(df_clean)
    
    energy_report = run_phase5_energy_optimization(df_clean)
    
    print("\nSaving reports...")
    with open("reports/energy_optimization_report.json", "w") as f:
        json.dump(energy_report, f, indent=2, default=str)
    print("  - reports/energy_optimization_report.json")
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print("\nGenerated artifacts:")
    print("  - data/interim/turin_clean.parquet")
    print("  - data/processed/turin_features.parquet")
    print("  - models/random_forest.joblib")
    print("  - models/xgboost_model.joblib")
    print("  - reports/energy_optimization_report.json")
    print("\nNext steps:")
    print("  - Review phase outputs")
    print("  - Tune model hyperparameters")
    print("  - Deploy models to production")


if __name__ == "__main__":
    main()
