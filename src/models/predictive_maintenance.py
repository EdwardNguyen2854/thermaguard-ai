import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def create_failure_labels(
    df: pd.DataFrame,
    target_col: str = "Power",
    threshold: float = 0.0,
    window_size: int = 24
) -> pd.DataFrame:
    """Create binary failure labels based on operational thresholds.
    
    Args:
        df: Input DataFrame
        target_col: Column to base failure detection on
        threshold: Threshold for failure detection
        window_size: Window for considering failures
        
    Returns:
        DataFrame with failure labels
    """
    df = df.copy()
    
    if target_col in df.columns:
        df['is_failure'] = (df[target_col] <= threshold).astype(int)
        
        df['failure_window'] = df['is_failure'].rolling(
            window=window_size, min_periods=1
        ).sum()
        
        df['failure_imminent'] = (df['failure_window'] > 0).astype(int)
        
        df['rul'] = 0
        for i in range(len(df) - 1, -1, -1):
            if df.iloc[i]['is_failure'] == 1:
                df.loc[df.index[i], 'rul'] = window_size
            elif i < len(df) - 1:
                df.loc[df.index[i], 'rul'] = max(0, df.iloc[i + 1]['rul'] - 1)
    
    return df


def balance_classes(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "undersample"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Balance classes for training.
    
    Args:
        X: Features
        y: Labels
        method: Balancing method
        
    Returns:
        Balanced X and y
    """
    if method == "undersample":
        minority_class = y.value_counts().idxmin()
        minority_indices = y[y == minority_class].index
        majority_indices = y[y != minority_class].sample(n=len(minority_indices)).index
        balanced_indices = minority_indices.union(majority_indices)
        return X.loc[balanced_indices], y.loc[balanced_indices]
    
    return X, y


def prepare_train_test_split(
    df: pd.DataFrame,
    target_col: str = "failure_imminent",
    test_size: float = 0.2,
    feature_cols: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare train/test split maintaining time order.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        test_size: Test set proportion
        feature_cols: Features to use
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if feature_cols is None:
        exclude = [target_col, 'rul', 'is_failure', 'failure_window', 'Timestamp']
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    df_clean = df.dropna(subset=feature_cols + [target_col])
    
    split_idx = int(len(df_clean) * (1 - test_size))
    
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    **kwargs
) -> RandomForestClassifier:
    """Train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        **kwargs: Additional parameters
        
    Returns:
        Trained model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs
) -> 'XGBClassifier':
    """Train XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        **kwargs: Additional parameters
        
    Returns:
        Trained model
    """
    try:
        from xgboost import XGBClassifier
        
        scale_pos = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            **kwargs
        )
        model.fit(X_train, y_train)
        return model
    except ImportError:
        warnings.warn("XGBoost not available, falling back to RandomForest")
        return train_random_forest(X_train, y_train)


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs
) -> 'LGBMClassifier':
    """Train LightGBM classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        **kwargs: Additional parameters
        
    Returns:
        Trained model
    """
    try:
        from lightgbm import LGBMClassifier
        
        model = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42,
            verbose=-1,
            **kwargs
        )
        model.fit(X_train, y_train)
        return model
    except ImportError:
        warnings.warn("LightGBM not available, falling back to RandomForest")
        return train_random_forest(X_train, y_train)


def evaluate_classifier(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """Evaluate classification model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        "accuracy": float((y_pred == y_test).mean()),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    
    if y_pred_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
        except:
            pass
    
    return metrics


def time_series_cross_validate(
    model_fn,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> Dict:
    """Time series cross-validation.
    
    Args:
        model_fn: Function that creates and returns a model
        X: Features
        y: Labels
        n_splits: Number of CV splits
        
    Returns:
        Cross-validation results
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = model_fn()
        model.fit(X_tr, y_tr)
        
        y_pred = model.predict(X_val)
        score = (y_pred == y_val).mean()
        scores.append(score)
    
    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "scores": scores
    }


def get_feature_importance(
    model,
    feature_names: list,
    top_n: int = 20
) -> pd.DataFrame:
    """Get feature importance from model.
    
    Args:
        model: Trained model
        feature_names: Feature names
        top_n: Top N features to return
        
    Returns:
        DataFrame with feature importances
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        return pd.DataFrame()
    
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return df_importance.head(top_n)


def save_model(model, filepath: str) -> None:
    """Save model to file.
    
    Args:
        model: Trained model
        filepath: Output path
    """
    joblib.dump(model, filepath)


def load_model(filepath: str):
    """Load model from file.
    
    Args:
        filepath: Model path
        
    Returns:
        Loaded model
    """
    return joblib.load(filepath)


def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: list = None
) -> Dict[str, object]:
    """Train ensemble of models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        models: List of model functions
        
    Returns:
        Dictionary of trained models
    """
    if models is None:
        models = [
            ("rf", lambda: train_random_forest(X_train, y_train)),
            ("xgb", lambda: train_xgboost_model(X_train, y_train)),
            ("lgb", lambda: train_lightgbm_model(X_train, y_train))
        ]
    
    trained = {}
    for name, model_fn in models:
        try:
            trained[name] = model_fn()
        except Exception as e:
            warnings.warn(f"Failed to train {name}: {e}")
    
    return trained


def ensemble_predict(models: Dict, X: pd.DataFrame) -> np.ndarray:
    """Make ensemble predictions.
    
    Args:
        models: Dictionary of trained models
        X: Features
        
    Returns:
        Ensemble predictions (averaged probabilities)
    """
    predictions = []
    
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
    
    if predictions:
        return np.mean(predictions, axis=0)
    
    return np.zeros(len(X))
