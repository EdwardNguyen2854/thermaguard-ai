import pandas as pd
import numpy as np
from typing import List, Optional, Dict


def add_time_features(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp"
) -> pd.DataFrame:
    """Add time-based features from timestamp.
    
    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column
        
    Returns:
        DataFrame with time features
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['day_of_month'] = df[timestamp_col].dt.day
    df['week_of_year'] = df[timestamp_col].dt.isocalendar().week.astype(int)
    df['month'] = df[timestamp_col].dt.month
    df['quarter'] = df[timestamp_col].dt.quarter
    df['year'] = df[timestamp_col].dt.year
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18) & (df['day_of_week'] < 5)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    df['is_heating_season'] = ((df['month'] >= 10) | (df['month'] <= 4)).astype(int)
    df['is_cooling_season'] = ((df['month'] >= 6) & (df['month'] <= 9)).astype(int)
    
    return df


def add_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [4, 24, 96],
    prefix: str = "roll"
) -> pd.DataFrame:
    """Add rolling window statistics.
    
    Args:
        df: Input DataFrame
        columns: Columns to compute rolling stats
        windows: Window sizes (in data points, e.g., 4=1h, 96=24h for 15min data)
        prefix: Prefix for new column names
        
    Returns:
        DataFrame with rolling features
    """
    df = df.copy()
    
    for col in columns:
        for window in windows:
            df[f'{prefix}_{col}_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{prefix}_{col}_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
            df[f'{prefix}_{col}_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
            df[f'{prefix}_{col}_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
    
    return df


def add_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int] = [4, 24, 96],
    prefix: str = "lag"
) -> pd.DataFrame:
    """Add lag features.
    
    Args:
        df: Input DataFrame
        columns: Columns to create lags for
        lags: Lag periods (in data points)
        prefix: Prefix for new column names
        
    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    
    for col in columns:
        for lag in lags:
            df[f'{prefix}_{col}_{lag}'] = df[col].shift(lag)
            df[f'{prefix}_{col}_diff_{lag}'] = df[col] - df[col].shift(lag)
    
    return df


def add_ema_features(
    df: pd.DataFrame,
    columns: List[str],
    spans: List[int] = [12, 24, 48],
    prefix: str = "ema"
) -> pd.DataFrame:
    """Add exponential moving average features.
    
    Args:
        df: Input DataFrame
        columns: Columns to compute EMAs
        spans: EMA span values
        prefix: Prefix for new column names
        
    Returns:
        DataFrame with EMA features
    """
    df = df.copy()
    
    for col in columns:
        for span in spans:
            df[f'{prefix}_{col}_{span}'] = df[col].ewm(span=span, adjust=False).mean()
    
    return df


def add_rate_of_change(
    df: pd.DataFrame,
    columns: List[str],
    periods: List[int] = [4, 24],
    prefix: str = "roc"
) -> pd.DataFrame:
    """Add rate of change features.
    
    Args:
        df: Input DataFrame
        columns: Columns to compute rate of change
        periods: Periods for rate calculation
        prefix: Prefix for new column names
        
    Returns:
        DataFrame with rate of change features
    """
    df = df.copy()
    
    for col in columns:
        for period in periods:
            df[f'{prefix}_{col}_{period}'] = df[col].pct_change(periods=period)
    
    return df


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add HVAC domain-specific features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with domain features
    """
    df = df.copy()
    
    if 'T_Supply' in df.columns and 'T_Return' in df.columns:
        df['temp_difference'] = df['T_Return'] - df['T_Supply']
        df['temp_efficiency'] = df['T_Supply'] / df['T_Return'].replace(0, np.nan)
    
    if 'T_Supply' in df.columns and 'T_Outdoor' in df.columns:
        df['temp_lift'] = df['T_Supply'] - df['T_Outdoor']
    
    if 'RH_Supply' in df.columns and 'RH_Return' in df.columns:
        df['rh_difference'] = df['RH_Return'] - df['RH_Supply']
    
    if 'T_Saturation' in df.columns and 'T_Supply' in df.columns:
        df['saturation_margin'] = df['T_Saturation'] - df['T_Supply']
    
    if 'SP_Return' in df.columns and 'T_Return' in df.columns:
        df['setpoint_deviation'] = df['T_Return'] - df['SP_Return']
    
    if 'Energy' in df.columns:
        df['energy_rate'] = df['Energy'].diff().fillna(0)
        df['energy_rate'] = df['energy_rate'].clip(lower=0)
    
    if 'Power' in df.columns:
        df['power_state'] = (df['Power'] > 0).astype(int)
        groups = (df['power_state'] != df['power_state'].shift()).cumsum()
        df['power_on_duration'] = df.groupby(groups).cumcount() * df['power_state']
    
    return df


def create_crossover_features(
    df: pd.DataFrame,
    column: str,
    short_window: int = 4,
    long_window: int = 24,
    prefix: str = "ma"
) -> pd.DataFrame:
    """Create moving average crossover features.
    
    Args:
        df: Input DataFrame
        column: Column to compute crossovers
        short_window: Short MA window
        long_window: Long MA window
        prefix: Prefix for column names
        
    Returns:
        DataFrame with crossover features
    """
    df = df.copy()
    
    short_ma = df[column].rolling(window=short_window, min_periods=1).mean()
    long_ma = df[column].rolling(window=long_window, min_periods=1).mean()
    
    df[f'{prefix}_{column}_short_{short_window}'] = short_ma
    df[f'{prefix}_{column}_long_{long_window}'] = long_ma
    df[f'{prefix}_{column}_crossover_{short_window}_{long_window}'] = short_ma - long_ma
    
    return df


def engineer_features(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    numeric_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Apply all feature engineering steps.
    
    Args:
        df: Input DataFrame
        timestamp_col: Timestamp column name
        numeric_cols: Numeric columns to process
        
    Returns:
        DataFrame with all engineered features
    """
    df = df.copy()
    
    df = add_time_features(df, timestamp_col)
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != timestamp_col]
    
    df = add_rolling_features(df, numeric_cols, windows=[4, 24, 96])
    df = add_lag_features(df, numeric_cols, lags=[4, 24, 96])
    df = add_ema_features(df, numeric_cols, spans=[12, 24, 48])
    df = add_rate_of_change(df, numeric_cols, periods=[4, 24])
    df = add_domain_features(df)
    
    for col in numeric_cols[:3]:
        df = create_crossover_features(df, col, short_window=4, long_window=24)
    
    return df


def get_feature_importance_summary(df: pd.DataFrame, target_col: str) -> Dict:
    """Get summary of feature statistics.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        
    Returns:
        Dictionary with feature statistics
    """
    feature_cols = [c for c in df.columns if c != target_col]
    
    summary = {
        "total_features": len(feature_cols),
        "feature_names": feature_cols,
        "missing_counts": {},
        "variance": {}
    }
    
    for col in feature_cols:
        summary["missing_counts"][col] = df[col].isna().sum()
        if df[col].dtype in [np.number64, np.float64]:
            summary["variance"][col] = float(df[col].var())
    
    return summary
