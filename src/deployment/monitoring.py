import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import joblib


class ModelMonitor:
    """Monitor model performance and detect drift."""
    
    def __init__(self, models_dir: str = "models", data_dir: str = "data"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.predictions_log = []
        self.drift_threshold = 0.1
        
    def log_prediction(
        self,
        timestamp: str,
        features: Dict,
        prediction: int,
        probability: float,
        actual: Optional[int] = None
    ):
        """Log a prediction for monitoring."""
        entry = {
            "timestamp": timestamp,
            "features": features,
            "prediction": prediction,
            "probability": probability,
            "actual": actual,
            "logged_at": datetime.now().isoformat()
        }
        self.predictions_log.append(entry)
        
    def calculate_drift_score(self, new_data: pd.DataFrame, reference_data: pd.DataFrame) -> float:
        """Calculate data drift score between new and reference data."""
        drift_scores = []
        
        for col in new_data.select_dtypes(include=[np.number]).columns:
            if col in reference_data.columns:
                new_mean = new_data[col].mean()
                ref_mean = reference_data[col].mean()
                ref_std = reference_data[col].std()
                
                if ref_std > 0:
                    drift = abs(new_mean - ref_mean) / ref_std
                    drift_scores.append(drift)
        
        return np.mean(drift_scores) if drift_scores else 0.0
    
    def check_model_degradation(
        self,
        predictions: List[int],
        actuals: List[int]
    ) -> Dict:
        """Check if model performance has degraded."""
        if not predictions or not actuals:
            return {"status": "unknown", "message": "No data"}
        
        accuracy = np.mean(np.array(predictions) == np.array(actuals))
        
        return {
            "status": "degraded" if accuracy < 0.8 else "healthy",
            "accuracy": float(accuracy),
            "threshold": 0.8,
            "message": f"Model accuracy: {accuracy:.2%}"
        }
    
    def generate_performance_report(self) -> Dict:
        """Generate performance monitoring report."""
        if not self.predictions_log:
            return {"status": "no_data", "message": "No predictions logged"}
        
        predictions_with_actual = [
            p for p in self.predictions_log if p.get("actual") is not None
        ]
        
        if not predictions_with_actual:
            return {
                "status": "waiting_for_labels",
                "total_predictions": len(self.predictions_log)
            }
        
        preds = [p["prediction"] for p in predictions_with_actual]
        actuals = [p["actual"] for p in predictions_with_actual]
        
        accuracy = np.mean(np.array(preds) == np.array(actuals))
        
        return {
            "status": "healthy" if accuracy > 0.8 else "degraded",
            "total_predictions": len(self.predictions_log),
            "labeled_predictions": len(predictions_with_actual),
            "accuracy": float(accuracy),
            "drift_score": 0.0,
            "generated_at": datetime.now().isoformat()
        }


class DataDriftDetector:
    """Detect data drift in incoming sensor data."""
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        self.reference_stats = {}
        if reference_data is not None:
            self.fit(reference_data)
    
    def fit(self, reference_data: pd.DataFrame):
        """Fit on reference data."""
        for col in reference_data.select_dtypes(include=[np.number]).columns:
            self.reference_stats[col] = {
                "mean": reference_data[col].mean(),
                "std": reference_data[col].std(),
                "min": reference_data[col].min(),
                "max": reference_data[col].max()
            }
    
    def detect(self, new_data: pd.DataFrame, threshold: float = 3.0) -> Dict:
        """Detect drift in new data."""
        drift_detected = {}
        
        for col in self.reference_stats:
            if col in new_data.columns:
                ref = self.reference_stats[col]
                new_mean = new_data[col].mean()
                
                if ref["std"] > 0:
                    z_score = abs(new_mean - ref["mean"]) / ref["std"]
                    drift_detected[col] = {
                        "z_score": float(z_score),
                        "drifted": z_score > threshold,
                        "reference_mean": float(ref["mean"]),
                        "new_mean": float(new_mean)
                    }
        
        total_drifted = sum(1 for v in drift_detected.values() if v["drifted"])
        
        return {
            "drift_detected": total_drifted > 0,
            "drifted_features": total_drifted,
            "total_features": len(drift_detected),
            "details": drift_detected,
            "checked_at": datetime.now().isoformat()
        }


class AlertManager:
    """Manage and track alerts."""
    
    def __init__(self):
        self.alerts = []
        self.alert_cooldown = timedelta(hours=1)
        self.last_alert_time = {}
        
    def create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Create a new alert."""
        now = datetime.now()
        
        if alert_type in self.last_alert_time:
            if now - self.last_alert_time[alert_type] < self.alert_cooldown:
                return {"status": "suppressed", "message": "Alert in cooldown"}
        
        alert = {
            "id": len(self.alerts) + 1,
            "type": alert_type,
            "severity": severity,
            "message": message,
            "metadata": metadata or {},
            "timestamp": now.isoformat(),
            "acknowledged": False
        }
        
        self.alerts.append(alert)
        self.last_alert_time[alert_type] = now
        
        return alert
    
    def get_active_alerts(
        self,
        min_severity: str = "low",
        limit: int = 50
    ) -> List[Dict]:
        """Get active (unacknowledged) alerts."""
        severity_order = {"low": 1, "medium": 2, "high": 3}
        min_level = severity_order.get(min_severity, 1)
        
        active = [
            a for a in self.alerts 
            if not a["acknowledged"] and severity_order.get(a["severity"], 0) >= min_level
        ]
        
        return sorted(active, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def acknowledge_alert(self, alert_id: int) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_at"] = datetime.now().isoformat()
                return True
        return False
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alerts."""
        total = len(self.alerts)
        acknowledged = sum(1 for a in self.alerts if a["acknowledged"])
        
        by_severity = {}
        for alert in self.alerts:
            sev = alert["severity"]
            by_severity[sev] = by_severity.get(sev, 0) + 1
        
        return {
            "total_alerts": total,
            "active": total - acknowledged,
            "acknowledged": acknowledged,
            "by_severity": by_severity
        }


class PipelineMonitor:
    """Monitor end-to-end pipeline health."""
    
    def __init__(self):
        self.pipeline_runs = []
        
    def log_pipeline_run(
        self,
        pipeline_name: str,
        status: str,
        duration_seconds: float,
        metrics: Optional[Dict] = None
    ):
        """Log a pipeline execution."""
        run = {
            "pipeline": pipeline_name,
            "status": status,
            "duration_seconds": duration_seconds,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat()
        }
        self.pipeline_runs.append(run)
        
    def get_pipeline_health(self, pipeline_name: str) -> Dict:
        """Get health status for a pipeline."""
        runs = [r for r in self.pipeline_runs if r["pipeline"] == pipeline_name]
        
        if not runs:
            return {"status": "unknown", "message": "No runs recorded"}
        
        recent_runs = runs[-10:]
        success_rate = sum(1 for r in recent_runs if r["status"] == "success") / len(recent_runs)
        avg_duration = np.mean([r["duration_seconds"] for r in recent_runs])
        
        return {
            "status": "healthy" if success_rate > 0.9 else "degraded",
            "recent_runs": len(recent_runs),
            "success_rate": float(success_rate),
            "avg_duration_seconds": float(avg_duration),
            "last_run": recent_runs[-1]["timestamp"] if recent_runs else None
        }


def create_monitoring_report() -> Dict:
    """Create comprehensive monitoring report."""
    monitor = ModelMonitor()
    alert_manager = AlertManager()
    pipeline_monitor = PipelineMonitor()
    
    return {
        "generated_at": datetime.now().isoformat(),
        "model_performance": monitor.generate_performance_report(),
        "alerts": alert_manager.get_alert_summary(),
        "pipelines": {
            "training": pipeline_monitor.get_pipeline_health("training"),
            "inference": pipeline_monitor.get_pipeline_health("inference")
        }
    }
