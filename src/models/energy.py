import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional
import warnings


def profile_energy_consumption(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    power_col: str = "Power"
) -> Dict:
    """Profile energy consumption patterns.
    
    Args:
        df: Input DataFrame
        timestamp_col: Timestamp column
        power_col: Power consumption column
        
    Returns:
        Dictionary with consumption profiles
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['month'] = df[timestamp_col].dt.month
    
    hourly_consumption = df.groupby('hour')[power_col].agg(['mean', 'sum', 'std'])
    daily_consumption = df.groupby('day_of_week')[power_col].agg(['mean', 'sum'])
    monthly_consumption = df.groupby('month')[power_col].agg(['mean', 'sum'])
    
    on_hours = (df[power_col] > 0).sum()
    total_hours = len(df)
    duty_cycle = (on_hours / total_hours) * 100
    
    return {
        "hourly_profile": hourly_consumption.to_dict(),
        "daily_profile": daily_consumption.to_dict(),
        "monthly_profile": monthly_consumption.to_dict(),
        "duty_cycle_percent": duty_cycle,
        "total_energy": df[power_col].sum(),
        "average_power": df[power_col].mean(),
        "peak_power": df[power_col].max()
    }


def calculate_efficiency_metrics(
    df: pd.DataFrame,
    power_col: str = "Power",
    temp_col: str = "T_Supply",
    outdoor_temp: str = "T_Outdoor"
) -> pd.DataFrame:
    """Calculate HVAC efficiency metrics.
    
    Args:
        df: Input DataFrame
        power_col: Power consumption column
        temp_col: Supply temperature column
        outdoor_temp: Outdoor temperature column
        
    Returns:
        DataFrame with efficiency metrics
    """
    df = df.copy()
    
    if temp_col in df.columns and outdoor_temp in df.columns:
        df['temp_lift'] = df[temp_col] - df[outdoor_temp]
        df.loc[df['temp_lift'] == 0, 'temp_lift'] = 0.001
    
    if power_col in df.columns and temp_col in df.columns:
        df['power_per_degree'] = df[power_col] / df['temp_lift'].abs()
    
    if temp_col in df.columns and outdoor_temp in df.columns:
        ideal_cop = 1 / (1 - (df[temp_col] - df[outdoor_temp]) / (df[temp_col] + 273.15))
        ideal_cop = ideal_cop.replace([np.inf, -np.inf], np.nan)
        df['efficiency_ratio'] = df['temp_lift'].abs() / (df[power_col] + 0.001)
    
    return df


def detect_consumption_anomalies(
    df: pd.DataFrame,
    power_col: str = "Power",
    threshold_std: float = 2.0
) -> pd.DataFrame:
    """Detect abnormal energy consumption patterns.
    
    Args:
        df: Input DataFrame
        power_col: Power column
        threshold_std: Standard deviation threshold
        
    Returns:
        DataFrame with anomaly flags
    """
    df = df.copy()
    
    df['power_rolling_mean'] = df[power_col].rolling(window=24, min_periods=1).mean()
    df['power_rolling_std'] = df[power_col].rolling(window=24, min_periods=1).std()
    
    df['power_zscore'] = (
        (df[power_col] - df['power_rolling_mean']) / 
        df['power_rolling_std'].replace(0, 1)
    )
    
    df['consumption_anomaly'] = (np.abs(df['power_zscore']) > threshold_std).astype(int)
    
    return df


def build_baseline_model(
    df: pd.DataFrame,
    features: List[str],
    target: str = "Power"
) -> Tuple['LinearRegression', Dict]:
    """Build baseline consumption model.
    
    Args:
        df: Training DataFrame
        features: Feature columns
        target: Target column
        
    Returns:
        Tuple of (trained model, metrics)
    """
    df_clean = df.dropna(subset=features + [target])
    X = df_clean[features]
    y = df_clean[target]
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    metrics = {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2_score": r2
    }
    
    return model, metrics


def predict_energy_demand(
    model,
    df: pd.DataFrame,
    features: List[str]
) -> np.ndarray:
    """Predict energy demand.
    
    Args:
        model: Trained model
        df: Input DataFrame
        features: Feature columns
        
    Returns:
        Predictions
    """
    X = df[features].fillna(0)
    return model.predict(X)


def recommend_setpoints(
    df: pd.DataFrame,
    outdoor_temp_col: str = "T_Outdoor",
    current_setpoint: str = "SP_Return",
    target_temp: float = 20.0
) -> Dict:
    """Recommend optimal temperature setpoints.
    
    Args:
        df: Historical DataFrame
        outdoor_temp_col: Outdoor temperature column
        current_setpoint: Current setpoint column
        target_temp: Target indoor temperature
        
    Returns:
        Dictionary with recommendations
    """
    df_clean = df.dropna(subset=[outdoor_temp_col, current_setpoint])
    
    X = df_clean[[outdoor_temp_col]]
    y = df_clean[current_setpoint]
    
    model = LinearRegression()
    model.fit(X, y)
    
    outdoor_range = np.linspace(
        df[outdoor_temp_col].min(),
        df[outdoor_temp_col].max(),
        10
    ).reshape(-1, 1)
    
    recommended_setpoints = model.predict(outdoor_range)
    
    return {
        "model_coefficients": {
            "intercept": model.intercept_,
            "slope": model.coef_[0]
        },
        "recommendations": {
            f"outdoor_{temp:.1f}C": float(setpoint) 
            for temp, setpoint in zip(outdoor_range.flatten(), recommended_setpoints)
        }
    }


def optimize_schedule(
    df: pd.DataFrame,
    power_col: str = "Power",
    timestamp_col: str = "Timestamp"
) -> Dict:
    """Suggest schedule optimizations.
    
    Args:
        df: Input DataFrame
        power_col: Power column
        timestamp_col: Timestamp column
        
    Returns:
        Optimization recommendations
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    df['hour'] = df[timestamp_col].dt.hour
    
    hourly_usage = df.groupby('hour')[power_col].mean()
    
    peak_hours = hourly_usage[hourly_usage > hourly_usage.quantile(0.75)].index.tolist()
    off_peak_hours = hourly_usage[hourly_usage < hourly_usage.quantile(0.25)].index.tolist()
    
    recommendations = {
        "peak_hours_to_avoid": peak_hours,
        "optimal_off_peak_hours": off_peak_hours,
        "shiftable_load_percent": float(
            df[df['hour'].isin(peak_hours)][power_col].sum() / 
            df[power_col].sum() * 100
        ) if df[power_col].sum() > 0 else 0
    }
    
    return recommendations


def estimate_savings_potential(
    df: pd.DataFrame,
    baseline_power: float,
    optimized_power: float,
    hours_per_day: float = 24,
    electricity_cost: float = 0.12
) -> Dict:
    """Estimate potential energy savings.
    
    Args:
        df: DataFrame with data
        baseline_power: Baseline power consumption
        optimized_power: Optimized power consumption
        hours_per_day: Operating hours
        electricity_cost: Cost per kWh
        
    Returns:
        Savings estimates
    """
    daily_savings_kwh = (baseline_power - optimized_power) * hours_per_day
    annual_savings_kwh = daily_savings_kwh * 365
    
    daily_cost_baseline = baseline_power * hours_per_day * electricity_cost
    daily_cost_optimized = optimized_power * hours_per_day * electricity_cost
    daily_savings_cost = daily_cost_baseline - daily_cost_optimized
    
    return {
        "daily_savings_kwh": float(daily_savings_kwh),
        "annual_savings_kwh": float(annual_savings_kwh),
        "daily_savings_cost": float(daily_savings_cost),
        "annual_savings_cost": float(daily_savings_cost * 365),
        "savings_percentage": float(
            (baseline_power - optimized_power) / baseline_power * 100
        ) if baseline_power > 0 else 0
    }


def generate_optimization_report(
    df: pd.DataFrame,
    power_col: str = "Power"
) -> Dict:
    """Generate comprehensive energy optimization report.
    
    Args:
        df: Input DataFrame
        power_col: Power column
        
    Returns:
        Complete optimization report
    """
    consumption_profile = profile_energy_consumption(df, power_col=power_col)
    
    df_with_efficiency = calculate_efficiency_metrics(df, power_col=power_col)
    
    schedule_optimization = optimize_schedule(df, power_col=power_col)
    
    avg_power = df[power_col].mean()
    optimized_power = avg_power * 0.85
    savings = estimate_savings_potential(df, avg_power, optimized_power)
    
    return {
        "consumption_profile": consumption_profile,
        "schedule_optimization": schedule_optimization,
        "estimated_savings": savings,
        "recommendations": [
            f"Shift high-power operations to off-peak hours: {schedule_optimization.get('optimal_off_peak_hours', [])}",
            f"Potential annual savings: ${savings.get('annual_savings_cost', 0):.2f}",
            f"Reduce duty cycle by optimizing setpoints during {consumption_profile.get('duty_cycle_percent', 0):.1f}% idle time"
        ]
    }
