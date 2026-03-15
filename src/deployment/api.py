from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from datetime import datetime
import os

app = FastAPI(
    title="ThermaGuard AI API",
    description="Industrial ML platform for predictive maintenance, anomaly detection, and energy optimization",
    version="1.0.0"
)

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent.parent / "data"

rf_model = None
xgb_model = None
if_model = None
scaler = None
feature_cols = None


def load_models():
    """Load trained models on startup."""
    global rf_model, xgb_model, if_model, scaler, feature_cols
    
    try:
        rf_model = joblib.load(MODELS_DIR / "random_forest.joblib")
        xgb_model = joblib.load(MODELS_DIR / "xgboost_model.joblib")
        
        feature_df = pd.read_parquet(DATA_DIR / "processed" / "turin_features.parquet", 
                                      columns=["Timestamp", "T_Supply", "T_Return", "Power"])
        exclude = ['failure_imminent', 'rul', 'is_failure', 'failure_window', 'Timestamp']
        feature_cols = [c for c in feature_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(feature_df[c])]
        
        print("Models loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load models: {e}")


class SensorData(BaseModel):
    """Sensor data input schema."""
    timestamp: str
    T_Supply: float = Field(..., ge=-10, le=50)
    T_Return: float = Field(..., ge=-10, le=50)
    SP_Return: float = Field(..., ge=-10, le=50)
    T_Saturation: float = Field(..., ge=-10, le=50)
    T_Outdoor: float = Field(..., ge=-30, le=60)
    RH_Supply: float = Field(..., ge=0, le=100)
    RH_Return: float = Field(..., ge=0, le=100)
    RH_Outdoor: float = Field(..., ge=0, le=100)
    Energy: float = Field(..., ge=0)
    Power: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    timestamp: str
    failure_prediction: int
    failure_probability: float
    anomaly_score: float
    severity: str
    recommendations: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    models_loaded: bool
    uptime_seconds: float

start_time = datetime.now()


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    load_models()


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "name": "ThermaGuard AI API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - start_time).total_seconds()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": rf_model is not None,
        "uptime_seconds": uptime
    }


@app.post("/predict/failure", response_model=PredictionResponse, tags=["Predictions"])
async def predict_failure(data: SensorData):
    """Predict equipment failure."""
    if rf_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features = {
            'T_Supply': data.T_Supply,
            'T_Return': data.T_Return,
            'SP_Return': data.SP_Return,
            'T_Saturation': data.T_Saturation,
            'T_Outdoor': data.T_Outdoor,
            'RH_Supply': data.RH_Supply,
            'RH_Return': data.RH_Return,
            'RH_Outdoor': data.RH_Outdoor,
            'Energy': data.Energy,
            'Power': data.Power
        }
        
        X = pd.DataFrame([features])
        
        prediction = rf_model.predict(X)[0]
        proba = rf_model.predict_proba(X)[0][1]
        
        recommendations = []
        if prediction == 1:
            recommendations.append("Immediate maintenance recommended")
            if data.Power > 3.0:
                recommendations.append("High power consumption detected - check compressor")
            if abs(data.T_Supply - data.T_Return) > 5:
                recommendations.append("Large temperature differential - inspect heat exchange")
        else:
            recommendations.append("Equipment operating normally")
            if proba > 0.3:
                recommendations.append("Monitor closely - approaching threshold")
        
        return {
            "timestamp": data.timestamp,
            "failure_prediction": int(prediction),
            "failure_probability": float(proba),
            "anomaly_score": float(proba),
            "severity": "high" if proba > 0.7 else "medium" if proba > 0.3 else "low",
            "recommendations": recommendations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/batch", tags=["Predictions"])
async def predict_batch(size: int = 10):
    """Get batch predictions from recent data."""
    if rf_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = pd.read_parquet(DATA_DIR / "processed" / "turin_features.parquet")
        df = df.ffill().fillna(0).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        exclude = ['failure_imminent', 'rul', 'is_failure', 'failure_window', 'Timestamp']
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        
        df_sample = df.tail(size)
        X = df_sample[feature_cols]
        
        predictions = rf_model.predict(X)
        probabilities = rf_model.predict_proba(X)[:, 1]
        
        results = []
        for i, row in df_sample.iterrows():
            results.append({
                "timestamp": str(row.get("Timestamp", "")),
                "prediction": int(predictions[df_sample.index.get_loc(i)]),
                "probability": float(probabilities[df_sample.index.get_loc(i)])
            })
        
        return {"predictions": results, "count": len(results)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["Models"])
async def model_info():
    """Get model information."""
    return {
        "models": {
            "random_forest": {
                "type": "RandomForestClassifier",
                "status": "loaded" if rf_model is not None else "not_loaded"
            },
            "xgboost": {
                "type": "XGBClassifier", 
                "status": "loaded" if xgb_model is not None else "not_loaded"
            }
        },
        "features_count": len(feature_cols) if feature_cols else 0,
        "data_path": str(DATA_DIR)
    }


@app.get("/stats", tags=["Statistics"])
async def get_statistics():
    """Get current system statistics."""
    try:
        df = pd.read_parquet(DATA_DIR / "interim" / "turin_clean.parquet")
        
        return {
            "data_points": len(df),
            "date_range": {
                "start": str(df["Timestamp"].min()),
                "end": str(df["Timestamp"].max())
            },
            "average_power": float(df["Power"].mean()),
            "max_power": float(df["Power"].max()),
            "duty_cycle": float((df["Power"] > 0).mean() * 100)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
