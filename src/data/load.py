import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union


def load_csv(
    filepath: Union[str, Path],
    parse_dates: bool = True,
    date_column: str = "Timestamp",
    sep: str = ";"
) -> pd.DataFrame:
    """Load HVAC sensor data from CSV file.
    
    Args:
        filepath: Path to CSV file
        parse_dates: Whether to parse timestamp column
        date_column: Name of the timestamp column
        sep: Column separator
        
    Returns:
        DataFrame with loaded data
    """
    df = pd.read_csv(filepath, sep=sep)
    
    if parse_dates and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], utc=True)
        
    return df


def load_parquet(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load processed data from Parquet file.
    
    Args:
        filepath: Path to Parquet file
        
    Returns:
        DataFrame with loaded data
    """
    return pd.read_parquet(filepath)


def save_parquet(df: pd.DataFrame, filepath: Union[str, Path]) -> None:
    """Save DataFrame to Parquet file.
    
    Args:
        df: DataFrame to save
        filepath: Output path
    """
    df.to_parquet(filepath, index=False)


def get_raw_data_path(filename: str = "turin_hvac.csv") -> Path:
    """Get path to raw data file."""
    return Path(__file__).parent.parent.parent / "data" / "raw" / filename


def get_interim_data_path(filename: str = "turin_clean.parquet") -> Path:
    """Get path to interim data file."""
    return Path(__file__).parent.parent.parent / "data" / "interim" / filename


def get_processed_data_path(filename: str = "turin_features.parquet") -> Path:
    """Get path to processed data file."""
    return Path(__file__).parent.parent.parent / "data" / "processed" / filename


def load_hvac_data(
    use_cached: bool = True,
    data_path: Optional[str] = None
) -> pd.DataFrame:
    """Load HVAC sensor data with automatic caching.
    
    Args:
        use_cached: Use cached interim data if available
        data_path: Custom data path (optional)
        
    Returns:
        DataFrame with HVAC sensor data
    """
    if data_path:
        path = Path(data_path)
        if path.suffix == ".parquet":
            return load_parquet(path)
        return load_csv(path)
    
    interim_path = get_interim_data_path()
    raw_path = get_raw_data_path()
    
    if use_cached and interim_path.exists():
        return load_parquet(interim_path)
    
    return load_csv(raw_path)


def get_column_descriptions() -> dict:
    """Get descriptions for HVAC data columns."""
    return {
        "Timestamp": "Date and time of measurement",
        "T_Supply": "Supply air temperature (°C)",
        "T_Return": "Return air temperature (°C)",
        "SP_Return": "Return air temperature setpoint (°C)",
        "T_Saturation": "Saturation temperature (°C)",
        "T_Outdoor": "Outdoor temperature (°C)",
        "RH_Supply": "Supply air relative humidity (%)",
        "RH_Return": "Return air relative humidity (%)",
        "RH_Outdoor": "Outdoor relative humidity (%)",
        "Energy": "Cumulative energy consumption (kWh)",
        "Power": "Instantaneous power consumption (kW)"
    }
