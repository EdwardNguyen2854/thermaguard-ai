# AGENTS.md - ThermaGuard AI Development Guide

This file provides guidelines for agentic coding agents working in the ThermaGuard AI codebase.

## Project Overview

ThermaGuard AI is an industrial machine learning platform for predictive maintenance, anomaly detection, and energy optimization in HVAC systems. It processes HVAC sensor data (temperature, pressure, humidity, power consumption) to predict equipment failures and optimize energy usage.

## Project Structure

```
thermaguard-ai/
├── src/
│   ├── data/           # Data loading and cleaning (load.py, clean.py)
│   ├── features/       # Feature engineering (build_features.py)
│   ├── models/         # ML models (predictive_maintenance.py, anomaly.py, energy.py)
│   ├── analysis/       # EDA and visualizations
│   └── deployment/    # FastAPI and monitoring (api.py, monitoring.py)
├── data/
│   ├── raw/           # Original sensor data
│   ├── interim/       # Cleaned data
│   └── processed/    # Feature-engineered datasets
├── models/           # Trained model files (.joblib)
├── reports/          # Generated reports
├── run_pipeline.py   # Main pipeline entry point
└── dashboard.py      # Dashboard visualization
```

## Build/Lint/Test Commands

### Installation

```bash
pip install -r requirements.txt
```

### Running the Pipeline

```bash
python run_pipeline.py
```

### Running Tests

There are currently no tests in this project. To run a single test (once tests are added):

```bash
# Using pytest
pytest tests/ -v
pytest tests/test_file.py::test_function -v

# Using unittest
python -m unittest tests.test_module.TestClass.test_method -v
```

### Running the API Server

```bash
python -m uvicorn src.deployment.api:app --reload
```

### Web Interface

The web dashboard is available at `http://localhost:8000/`:

| Route | Description |
|-------|-------------|
| `/` | Dashboard with system overview and quick actions |
| `/data` | Data upload and management |
| `/training` | Model training configuration |
| `/predictions` | Single and batch predictions |
| `/analytics` | Data visualization and insights |

Frontend files are in `templates/` (HTML) and `static/` (CSS/JS).

### Linting (Recommended Tools)

Install linting tools:
```bash
pip install black flake8 isort mypy
```

Run linters:
```bash
# Format code
black src/
isort src/

# Check linting
flake8 src/ --max-line-length=100

# Type checking
mypy src/ --ignore-missing-imports
```

## Code Style Guidelines

### Import Organization

Organize imports in the following order with blank lines between groups:

1. Standard library imports
2. Third-party imports
3. Local/application imports

```python
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.load import load_hvac_data
from src.features.build_features import engineer_features
from src.models.predictive_maintenance import train_random_forest
```

### Type Hints

Use type hints for all function parameters and return values:

```python
def prepare_train_test_split(
    df: pd.DataFrame,
    target_col: str = "failure_imminent",
    test_size: float = 0.2,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
```

### Naming Conventions

- **Functions/variables**: snake_case (`train_random_forest`, `feature_cols`)
- **Classes**: PascalCase (`SensorData`, `PredictionResponse`)
- **Constants**: UPPER_SNAKE_CASE (`MODELS_DIR`, `DATA_DIR`)
- **Files**: snake_case (`predictive_maintenance.py`, `build_features.py`)

### Docstrings

Use Google-style docstrings for all public functions:

```python
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
```

### Error Handling

- Use try/except blocks with specific exception handling
- Catch exceptions at the appropriate level
- Provide meaningful error messages
- Use warnings for non-critical issues:

```python
try:
    from xgboost import XGBClassifier
    model = XGBClassifier(...)
    model.fit(X_train, y_train)
    return model
except ImportError:
    warnings.warn("XGBoost not available, falling back to RandomForest")
    return train_random_forest(X_train, y_train)
```

For API endpoints, use FastAPI's HTTPException:

```python
from fastapi import HTTPException

if rf_model is None:
    raise HTTPException(status_code=503, detail="Model not loaded")
```

### Data Validation

Use Pydantic models for API request/response validation:

```python
class SensorData(BaseModel):
    """Sensor data input schema."""
    timestamp: str
    T_Supply: float = Field(..., ge=-10, le=50)
    T_Return: float = Field(..., ge=-10, le=50)
    Power: float = Field(..., ge=0)
```

### DataFrame Handling

- Always copy DataFrames when modifying to avoid SettingWithCopyWarning
- Use `.copy()` at the start of functions that modify data
- Handle missing values explicitly (ffill, fillna, dropna)
- Replace infinity values:

```python
df = df.ffill().fillna(0)
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
```

### Model Development

- Use `random_state=42` for reproducibility
- Use `n_jobs=-1` for parallel processing where applicable
- Handle class imbalance with `class_weight='balanced'` or `scale_pos_weight`
- Always evaluate models and return metrics dictionaries
- Save models using joblib:

```python
import joblib

def save_model(model, filepath: str) -> None:
    joblib.dump(model, filepath)

def load_model(filepath: str):
    return joblib.load(filepath)
```

### API Development

- Use FastAPI with async/await for endpoints
- Include tags for endpoint organization
- Return appropriate HTTP status codes
- Use Pydantic models for request/response schemas

### File Paths

Use `pathlib.Path` for cross-platform path handling:

```python
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
```

### Logging and Output

- Use print statements for pipeline progress (simple approach)
- For production, use the logging module
- Print formatted output with section headers:

```python
print("=" * 60)
print("Phase 3: Predictive Maintenance Models")
print("=" * 60)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
```

## Data Schema

| Column | Description | Range |
|--------|-------------|-------|
| Timestamp | DateTime | 2019-10-15 onwards |
| T_Supply | Supply air temperature (°C) | -10 to 50 |
| T_Return | Return air temperature (°C) | -10 to 50 |
| SP_Return | Return setpoint (°C) | -10 to 50 |
| T_Saturation | Saturation temperature (°C) | -10 to 50 |
| T_Outdoor | Outdoor temperature (°C) | -30 to 60 |
| RH_Supply | Supply humidity (%) | 0-100 |
| RH_Return | Return humidity (%) | 0-100 |
| RH_Outdoor | Outdoor humidity (%) | 0-100 |
| Energy | Cumulative energy (kWh) | 0+ |
| Power | Instantaneous power (kW) | 0-10 |

## Key Dependencies

- **Data Processing**: pandas>=2.0.0, numpy>=1.24.0
- **Machine Learning**: scikit-learn>=1.3.0, xgboost>=2.0.0, lightgbm>=4.0.0
- **Deep Learning**: tensorflow>=2.14.0
- **Visualization**: matplotlib>=3.7.0, seaborn>=0.12.0, plotly>=5.18.0
- **API**: fastapi, uvicorn, pydantic

## Common Tasks

### Running a specific pipeline phase

Import and call the function directly from Python:
```python
from run_pipeline import run_phase3_predictive_maintenance
df_features = pd.read_parquet("data/processed/turin_features.parquet")
df_labeled, rf_model = run_phase3_predictive_maintenance(df_features)
```

### Adding a new model

1. Create a new file in `src/models/`
2. Follow the naming conventions (snake_case)
3. Include type hints and docstrings
4. Export functions from `src/models/__init__.py`
5. Update `run_pipeline.py` to include the new model

### Adding API endpoints

1. Add Pydantic models for request/response in `src/deployment/api.py`
2. Add new endpoint with appropriate tags
3. Handle errors with HTTPException
