import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from pathlib import Path


def plot_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot distributions of numeric columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to plot (None = all numeric)
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for i, col in enumerate(columns):
        axes[i].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[i].set_title(f'{col} Distribution')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: tuple = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot correlation matrix heatmap.
    
    Args:
        df: Input DataFrame
        columns: Columns to include (None = all numeric)
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr = df[columns].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax, square=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_time_series(
    df: pd.DataFrame,
    columns: List[str],
    timestamp_col: str = "Timestamp",
    n_points: int = 1000,
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot time series data.
    
    Args:
        df: Input DataFrame
        columns: Columns to plot
        timestamp_col: Timestamp column name
        n_points: Number of points to plot (for performance)
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    plot_df = df.copy()
    if len(plot_df) > n_points:
        plot_df = plot_df.iloc[::len(plot_df)//n_points]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in columns:
        ax.plot(plot_df[timestamp_col], plot_df[col], label=col, alpha=0.7)
    
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Value')
    ax.set_title('Time Series Overview')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_missing_values(
    df: pd.DataFrame,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot missing values heatmap.
    
    Args:
        df: Input DataFrame
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    missing = df.isnull()
    missing_pct = (missing.sum() / len(df)) * 100
    
    fig, ax = plt.subplots(figsize=figsize)
    missing_pct.plot(kind='bar', ax=ax, color='coral')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Missing Percentage (%)')
    ax.set_title('Missing Values by Column')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_daily_patterns(
    df: pd.DataFrame,
    columns: List[str],
    timestamp_col: str = "Timestamp",
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot hourly average patterns.
    
    Args:
        df: Input DataFrame
        columns: Columns to analyze
        timestamp_col: Timestamp column name
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    df = df.copy()
    df['hour'] = pd.to_datetime(df[timestamp_col]).dt.hour
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in columns:
        hourly_avg = df.groupby('hour')[col].mean()
        ax.plot(hourly_avg.index, hourly_avg.values, label=col, marker='o')
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Value')
    ax.set_title('Daily Pattern Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(24))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_eda_report(
    df: pd.DataFrame,
    output_dir: str = "reports/figures",
    timestamp_col: str = "Timestamp"
) -> dict:
    """Generate comprehensive EDA report.
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save figures
        timestamp_col: Timestamp column name
        
    Returns:
        Summary statistics dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    plot_distributions(df, numeric_cols, save_path=str(output_path / "distributions.png"))
    plot_correlation_matrix(df, numeric_cols, save_path=str(output_path / "correlation.png"))
    plot_missing_values(df, save_path=str(output_path / "missing.png"))
    plot_daily_patterns(df, numeric_cols[:5], timestamp_col, save_path=str(output_path / "daily_patterns.png"))
    
    summary = {
        "shape": df.shape,
        "date_range": {
            "start": df[timestamp_col].min() if timestamp_col in df.columns else None,
            "end": df[timestamp_col].max() if timestamp_col in df.columns else None
        },
        "numeric_columns": numeric_cols,
        "statistics": df[numeric_cols].describe().to_dict()
    }
    
    return summary
