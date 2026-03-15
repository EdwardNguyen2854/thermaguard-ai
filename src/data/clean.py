import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List, Optional


def validate_timestamp_continuity(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    expected_freq: str = "15min"
) -> pd.DataFrame:
    """Validate and ensure timestamp continuity.
    
    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column
        expected_freq: Expected frequency (e.g., '15min', '1h')
        
    Returns:
        DataFrame with continuous timestamps
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col)
    
    expected_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=expected_freq
    )
    
    df = df.reindex(expected_index)
    df.index.name = timestamp_col
    df = df.reset_index()
    
    return df


def handle_missing_values(
    df: pd.DataFrame,
    method: str = "forward",
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        method: Interpolation method ('forward', 'backward', 'linear', 'interpolate')
        columns: Columns to process (None = all numeric columns)
        
    Returns:
        DataFrame with filled missing values
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == "forward":
        df[columns] = df[columns].ffill()
    elif method == "backward":
        df[columns] = df[columns].bfill()
    elif method == "linear":
        df[columns] = df[columns].interpolate(method="linear")
    else:
        df[columns] = df[columns].interpolate(method=method)
    
    return df


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    multiplier: float = 1.5
) -> Tuple[pd.DataFrame, dict]:
    """Remove outliers using IQR method.
    
    Args:
        df: Input DataFrame
        columns: Columns to check for outliers (None = all numeric)
        multiplier: IQR multiplier for outlier threshold
        
    Returns:
        Tuple of (cleaned DataFrame, outlier info dict)
    """
    df = df.copy()
    outlier_info = {}
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {
            "count": len(outliers),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
        
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df, outlier_info


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Remove duplicate rows.
    
    Args:
        df: Input DataFrame
        subset: Columns to check for duplicates
        
    Returns:
        DataFrame without duplicates
    """
    return df.drop_duplicates(subset=subset)


def validate_sensor_ranges(
    df: pd.DataFrame,
    sensor_ranges: Optional[dict] = None
) -> dict:
    """Validate sensor readings are within expected ranges.
    
    Args:
        df: Input DataFrame
        sensor_ranges: Dict of column -> (min, max) ranges
        
    Returns:
        Validation results dictionary
    """
    if sensor_ranges is None:
        sensor_ranges = {
            "T_Supply": (-10, 50),
            "T_Return": (-10, 50),
            "T_Saturation": (-10, 50),
            "T_Outdoor": (-30, 60),
            "RH_Supply": (0, 100),
            "RH_Return": (0, 100),
            "RH_Outdoor": (0, 100),
            "Energy": (0, 1e6),
            "Power": (0, 1000)
        }
    
    results = {}
    for col, (min_val, max_val) in sensor_ranges.items():
        if col in df.columns:
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            results[col] = {
                "out_of_range_count": len(out_of_range),
                "min_allowed": min_val,
                "max_allowed": max_val
            }
    
    return results


def clean_hvac_data(
    df: pd.DataFrame,
    fill_method: str = "forward",
    remove_outliers: bool = True,
    ensure_continuity: bool = True
) -> pd.DataFrame:
    """Clean HVAC sensor data.
    
    Args:
        df: Raw HVAC DataFrame
        fill_method: Method to fill missing values
        remove_outliers: Whether to remove outliers
        ensure_continuity: Ensure timestamp continuity
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    
    initial_rows = len(df)
    df = remove_duplicates(df)
    
    if ensure_continuity and "Timestamp" in df.columns:
        df = validate_timestamp_continuity(df)
    
    df = handle_missing_values(df, method=fill_method)
    
    if remove_outliers:
        df, _ = remove_outliers_iqr(df)
    
    final_rows = len(df)
    
    return df


def get_data_quality_report(df: pd.DataFrame) -> dict:
    """Generate data quality report.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with quality metrics
    """
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": {},
        "statistics": {},
        "dtypes": df.dtypes.to_dict()
    }
    
    for col in df.columns:
        missing = df[col].isna().sum()
        missing_pct = (missing / len(df)) * 100
        report["missing_values"][col] = {
            "count": missing,
            "percentage": round(missing_pct, 2)
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            report["statistics"][col] = {
                "mean": round(df[col].mean(), 4),
                "std": round(df[col].std(), 4),
                "min": round(df[col].min(), 4),
                "max": round(df[col].max(), 4),
                "median": round(df[col].median(), 4)
            }
    
    return report
