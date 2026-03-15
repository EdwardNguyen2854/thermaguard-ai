# ThermaGuard AI

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/framework-sklearn%20%7C%20xgboost%20%7C%20fastapi-orange.svg)](https://)
[![Dataset](https://img.shields.io/badge/dataset-33.8K%20records-blueviolet.svg)](https://)

Industrial ML platform for predictive maintenance, anomaly detection, and energy optimization in HVAC systems.

## Overview

ThermaGuard AI uses machine learning to analyze HVAC sensor data—temperature, humidity, pressure, and power consumption—to:

- **Predict equipment failures** before they occur
- **Detect anomalies** in real-time operating conditions
- **Optimize energy consumption** and reduce operational costs

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Run the full pipeline

```bash
python run_pipeline.py
```

### Start the API server

```bash
python -m uvicorn src.deployment.api:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Model Performance

| Model | Accuracy | Task |
|-------|----------|------|
| Random Forest | 99.28% | Failure Prediction |
| XGBoost | 100.00% | Failure Prediction |
| Isolation Forest | — | Anomaly Detection |

### Key Metrics

- **Duty Cycle**: 48.21%
- **Peak Hours**: 6 AM – 11 AM
- **Off-Peak Hours**: 12 AM – 2 AM, 9 PM – 11 PM
- **Estimated Annual Savings**: $353.83

## Features

667 engineered features including:

- Time-based (hour, day, cyclical encoding, seasonal flags)
- Rolling statistics (mean, std, min, max; 1h, 6h, 24h windows)
- Lag features and differencing
- Domain-specific HVAC indicators

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict/failure` | POST | Predict equipment failure |
| `/predict/batch` | GET | Batch predictions |
| `/model/info` | GET | Model metadata |
| `/stats` | GET | System statistics |

## Data Schema

| Column | Description | Range |
|--------|-------------|-------|
| `Timestamp` | DateTime | 2019-10-15+ |
| `T_Supply` | Supply air temperature (°C) | -10 – 50 |
| `T_Return` | Return air temperature (°C) | -10 – 50 |
| `SP_Return` | Return setpoint (°C) | -10 – 50 |
| `T_Saturation` | Saturation temperature (°C) | -10 – 50 |
| `T_Outdoor` | Outdoor temperature (°C) | -30 – 60 |
| `RH_Supply` | Supply humidity (%) | 0 – 100 |
| `RH_Return` | Return humidity (%) | 0 – 100 |
| `RH_Outdoor` | Outdoor humidity (%) | 0 – 100 |
| `Energy` | Cumulative energy (kWh) | 0+ |
| `Power` | Instantaneous power (kW) | 0 – 10 |

## Dataset

| Attribute | Value |
|-----------|-------|
| Source | Turin HVAC System (Italy) |
| Raw Records | 33,888 |
| Clean Records | 25,691 |
| Time Range | October 2019 – ongoing |
| Frequency | 15-minute intervals |

The dataset contains sensor readings from an industrial HVAC system including temperature, humidity, and power consumption measurements.

## Usage Example

```python
from src.models.predictive_maintenance import create_failure_labels, train_random_forest
import pandas as pd

# Load and label data
df = pd.read_parquet("data/processed/turin_features.parquet")
df_labeled = create_failure_labels(df, target_col="Power", threshold=0.0)

# Train model
rf_model = train_random_forest(X_train, y_train)

# Predict
prediction = rf_model.predict(X_test)
```

## License

MIT License — see [LICENSE](LICENSE) for details.
