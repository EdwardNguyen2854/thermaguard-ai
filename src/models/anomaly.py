import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from typing import Dict, List, Tuple, Optional
import warnings


def zscore_anomaly_detection(
    df: pd.DataFrame,
    columns: List[str],
    threshold: float = 3.0
) -> pd.DataFrame:
    """Detect anomalies using z-score method.
    
    Args:
        df: Input DataFrame
        columns: Columns to check
        threshold: Z-score threshold
        
    Returns:
        DataFrame with anomaly scores
    """
    df = df.copy()
    
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df[f'{col}_zscore'] = (df[col] - mean) / std
        df[f'{col}_is_anomaly_zscore'] = (np.abs(df[f'{col}_zscore']) > threshold).astype(int)
    
    df['zscore_anomaly_count'] = df[[f'{c}_is_anomaly_zscore' for c in columns]].sum(axis=1)
    
    return df


def iqr_anomaly_detection(
    df: pd.DataFrame,
    columns: List[str],
    multiplier: float = 1.5
) -> pd.DataFrame:
    """Detect anomalies using IQR method.
    
    Args:
        df: Input DataFrame
        columns: Columns to check
        multiplier: IQR multiplier
        
    Returns:
        DataFrame with anomaly flags
    """
    df = df.copy()
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        
        df[f'{col}_is_anomaly_iqr'] = ((df[col] < lower) | (df[col] > upper)).astype(int)
    
    df['iqr_anomaly_count'] = df[[f'{c}_is_anomaly_iqr' for c in columns]].sum(axis=1)
    
    return df


def moving_window_threshold(
    df: pd.DataFrame,
    columns: List[str],
    window_size: int = 24,
    std_multiplier: float = 3.0
) -> pd.DataFrame:
    """Detect anomalies using moving window thresholds.
    
    Args:
        df: Input DataFrame
        columns: Columns to check
        window_size: Rolling window size
        std_multiplier: Standard deviation multiplier
        
    Returns:
        DataFrame with anomaly flags
    """
    df = df.copy()
    
    for col in columns:
        rolling_mean = df[col].rolling(window=window_size, min_periods=1).mean()
        rolling_std = df[col].rolling(window=window_size, min_periods=1).std()
        
        df[f'{col}_is_anomaly_mw'] = (
            np.abs(df[col] - rolling_mean) > std_multiplier * rolling_std
        ).astype(int)
    
    df['mw_anomaly_count'] = df[[f'{c}_is_anomaly_mw' for c in columns]].sum(axis=1)
    
    return df


def train_isolation_forest(
    df: pd.DataFrame,
    columns: List[str],
    contamination: float = 0.1,
    random_state: int = 42
) -> Tuple[IsolationForest, StandardScaler]:
    """Train Isolation Forest model.
    
    Args:
        df: Training DataFrame
        columns: Features to use
        contamination: Expected proportion of anomalies
        random_state: Random seed
        
    Returns:
        Tuple of (trained model, scaler)
    """
    scaler = StandardScaler()
    X = df[columns].fillna(0)
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_scaled)
    
    return model, scaler


def isolation_forest_predict(
    model: IsolationForest,
    scaler: StandardScaler,
    df: pd.DataFrame,
    columns: List[str]
) -> pd.DataFrame:
    """Predict anomalies using Isolation Forest.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        df: Input DataFrame
        columns: Features
        
    Returns:
        DataFrame with predictions
    """
    df = df.copy()
    X = df[columns].fillna(0)
    X_scaled = scaler.transform(X)
    
    df['if_anomaly'] = model.predict(X_scaled)
    df['if_anomaly'] = (df['if_anomaly'] == -1).astype(int)
    df['if_score'] = model.decision_function(X_scaled)
    
    return df


def train_autoencoder(
    X: np.ndarray,
    encoding_dim: int = 32,
    epochs: int = 50,
    batch_size: int = 32
):
    """Train simple autoencoder for anomaly detection.
    
    Args:
        X: Training data
        encoding_dim: Encoding dimension
        epochs: Training epochs
        batch_size: Batch size
        
    Returns:
        Trained autoencoder
    """
    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense
        from tensorflow.keras.callbacks import EarlyStopping
        
        input_dim = X.shape[1]
        
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='relu')(encoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        early_stop = EarlyStopping(patience=5, restore_best_weights=True)
        
        autoencoder.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )
        
        return autoencoder
    except ImportError:
        warnings.warn("TensorFlow not available, skipping autoencoder")
        return None


def autoencoder_predict(
    autoencoder,
    X: np.ndarray,
    threshold: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """Predict anomalies using autoencoder reconstruction error.
    
    Args:
        autoencoder: Trained autoencoder
        X: Input data
        threshold: Anomaly threshold (auto-calculated if None)
        
    Returns:
        Tuple of (anomaly flags, threshold)
    """
    if autoencoder is None:
        return np.zeros(len(X)), 0.0
    
    X_pred = autoencoder.predict(X, verbose=0)
    mse = np.mean(np.power(X - X_pred, 2), axis=1)
    
    if threshold is None:
        threshold = np.percentile(mse, 95)
    
    anomalies = (mse > threshold).astype(int)
    
    return anomalies, threshold


def ensemble_anomaly_score(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """Combine multiple anomaly detection methods.
    
    Args:
        df: DataFrame with anomaly scores from multiple methods
        weights: Weights for each method
        
    Returns:
        DataFrame with combined scores
    """
    df = df.copy()
    
    if weights is None:
        weights = {
            'zscore': 0.3,
            'iqr': 0.3,
            'isolation_forest': 0.4
        }
    
    score_columns = []
    
    if 'zscore_anomaly_count' in df.columns:
        df['zscore_normalized'] = df['zscore_anomaly_count'] / df['zscore_anomaly_count'].max()
        score_columns.append('zscore_normalized')
    
    if 'iqr_anomaly_count' in df.columns:
        df['iqr_normalized'] = df['iqr_anomaly_count'] / df['iqr_anomaly_count'].max()
        score_columns.append('iqr_normalized')
    
    if 'if_score' in df.columns:
        df['if_normalized'] = (df['if_score'] - df['if_score'].min()) / (df['if_score'].max() - df['if_score'].min())
        score_columns.append('if_normalized')
    
    if score_columns:
        df['ensemble_score'] = 0
        for col, weight in weights.items():
            if col == 'zscore' and 'zscore_normalized' in df.columns:
                df['ensemble_score'] += weight * df['zscore_normalized']
            elif col == 'iqr' and 'iqr_normalized' in df.columns:
                df['ensemble_score'] += weight * df['iqr_normalized']
            elif col == 'isolation_forest' and 'if_normalized' in df.columns:
                df['ensemble_score'] += weight * df['if_normalized']
        
        df['ensemble_anomaly'] = (df['ensemble_score'] > 0.5).astype(int)
    
    return df


def classify_alert_severity(
    df: pd.DataFrame,
    score_column: str = 'ensemble_score'
) -> pd.DataFrame:
    """Classify anomaly alert severity.
    
    Args:
        df: DataFrame with anomaly scores
        score_column: Column containing scores
        
    Returns:
        DataFrame with severity classification
    """
    df = df.copy()
    
    if score_column in df.columns:
        df['severity'] = 'normal'
        df.loc[df[score_column] > 0.2, 'severity'] = 'low'
        df.loc[df[score_column] > 0.4, 'severity'] = 'medium'
        df.loc[df[score_column] > 0.7, 'severity'] = 'high'
    
    return df


def filter_alerts(
    df: pd.DataFrame,
    min_severity: str = 'low',
    cooldown: int = 4
) -> pd.DataFrame:
    """Filter and deduplicate alerts.
    
    Args:
        df: DataFrame with alerts
        min_severity: Minimum severity to include
        cooldown: Minimum samples between alerts
        
    Returns:
        Filtered DataFrame
    """
    severity_order = {'normal': 0, 'low': 1, 'medium': 2, 'high': 3}
    
    if 'severity' in df.columns:
        df['severity_level'] = df['severity'].map(severity_order)
        min_level = severity_order.get(min_severity, 1)
        df = df[df['severity_level'] >= min_level]
        
        df['alert_group'] = (df['severity_level'] > 0).astype(int)
        df['alert_group'] = df['alert_group'].groupby(
            (df['alert_group'] != df['alert_group'].shift()).cumsum()
        ).cumsum()
        
        df = df.groupby('alert_group').first().reset_index(drop=True)
    
    return df
