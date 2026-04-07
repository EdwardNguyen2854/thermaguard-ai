from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import io
import traceback

from src.data.load import load_hvac_data, load_csv
from src.data.clean import clean_hvac_data
from src.features.build_features import engineer_features
from src.models.predictive_maintenance import (
    create_failure_labels, balance_classes, prepare_train_test_split,
    train_random_forest, train_xgboost_model, train_lightgbm_model,
    evaluate_classifier, get_feature_importance, save_model as save_ml_model,
    train_ensemble, ensemble_predict
)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'thermaguard-secret-key-2024'

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
MODELS_DIR.mkdir(exist_ok=True)


@app.context_processor
def inject_functions():
    return {
        'enumerate': enumerate,
        'len': len,
        'str': str,
        'float': float,
        'int': int,
        'round': round
    }


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/data')
def data_page():
    """Data management page."""
    return render_template('data.html')


@app.route('/training')
def training_page():
    """Model training page."""
    return render_template('training.html')


@app.route('/predictions')
def predictions_page():
    """Predictions page."""
    return render_template('predictions.html')


@app.route('/analytics')
def analytics_page():
    """Analytics page."""
    return render_template('analytics.html')


@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    """Handle CSV file upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        df = load_csv(io.BytesIO(file.read()))
        session['data'] = df.to_json()
        
        return jsonify({
            'success': True,
            'rows': len(df),
            'columns': list(df.columns)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/load-sample-data')
def load_sample_data():
    """Load sample HVAC data."""
    try:
        df = load_hvac_data(use_cached=True)
        session['data'] = df.to_json()
        return jsonify({
            'success': True,
            'rows': len(df),
            'columns': list(df.columns)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-data-info')
def get_data_info():
    """Get information about loaded data."""
    try:
        data_json = session.get('data')
        if not data_json:
            return jsonify({'loaded': False})
        
        df = pd.read_json(data_json)
        
        return jsonify({
            'loaded': True,
            'rows': len(df),
            'columns': list(df.columns),
            'date_range': {
                'start': str(df['Timestamp'].min()) if 'Timestamp' in df.columns else 'N/A',
                'end': str(df['Timestamp'].max()) if 'Timestamp' in df.columns else 'N/A'
            },
            'missing_values': int(df.isnull().sum().sum())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-data-preview')
def get_data_preview():
    """Get data preview for table display."""
    try:
        data_json = session.get('data')
        if not data_json:
            return jsonify({'error': 'No data loaded'}), 400
        
        df = pd.read_json(data_json)
        
        return jsonify({
            'columns': list(df.columns),
            'rows': df.head(100).to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Train a new model."""
    try:
        data_json = session.get('data')
        if not data_json:
            return jsonify({'error': 'No data loaded'}), 400
        
        df = pd.read_json(data_json)
        config = request.get_json()
        
        model_type = config.get('model_type', 'random_forest')
        test_size = float(config.get('test_size', 0.2))
        failure_threshold = float(config.get('failure_threshold', 0.0))
        window_size = int(config.get('window_size', 24))
        balance_method = config.get('balance_method', 'none')
        
        df = create_failure_labels(df, target_col="Power", threshold=failure_threshold, window_size=window_size)
        
        feature_cols = [c for c in df.columns 
                       if c not in ['failure_imminent', 'rul', 'is_failure', 'failure_window', 'Timestamp'] 
                       and pd.api.types.is_numeric_dtype(df[c])]
        
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            df, target_col="failure_imminent", test_size=test_size, feature_cols=feature_cols
        )
        
        if balance_method == 'undersample':
            X_train, y_train = balance_classes(X_train, y_train, 'undersample')
        
        if model_type == 'random_forest':
            model = train_random_forest(
                X_train, y_train,
                n_estimators=int(config.get('n_estimators', 100)),
                max_depth=int(config.get('max_depth', 15))
            )
        elif model_type == 'xgboost':
            model = train_xgboost_model(
                X_train, y_train,
                n_estimators=int(config.get('n_estimators', 100)),
                max_depth=int(config.get('max_depth', 6)),
                learning_rate=float(config.get('learning_rate', 0.1))
            )
        elif model_type == 'lightgbm':
            model = train_lightgbm_model(
                X_train, y_train,
                n_estimators=int(config.get('n_estimators', 100)),
                max_depth=int(config.get('max_depth', 6)),
                learning_rate=float(config.get('learning_rate', 0.1))
            )
        else:
            models_dict = train_ensemble(X_train, y_train)
            model = models_dict
        
        X_test_clean = X_test.fillna(0).replace([np.inf, -np.inf], 0)
        
        if model_type == 'ensemble':
            probas = ensemble_predict(models_dict, X_test_clean)
            predictions = (probas > 0.5).astype(int)
            metrics = {
                'accuracy': float((predictions == y_test).mean()),
                'roc_auc': float(pd.Series(probas).sum())
            }
        else:
            metrics = evaluate_classifier(model, X_test_clean, y_test)
        
        model_name = f"{model_type}_{datetime.now().strftime('%H%M%S')}"
        model_path = MODELS_DIR / f"{model_name}.joblib"
        
        if model_type == 'ensemble':
            joblib.dump(models_dict, model_path)
        else:
            save_ml_model(model, str(model_path))
        
        session['trained_models'] = session.get('trained_models', {})
        session['trained_models'][model_name] = {
            'model': str(model_path),
            'model_type': model_type,
            'metrics': metrics,
            'feature_cols': feature_cols
        }
        session['current_model'] = model_name
        
        return jsonify({
            'success': True,
            'model_name': model_name,
            'metrics': metrics
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-trained-models')
def get_trained_models():
    """Get list of trained models."""
    models = session.get('trained_models', {})
    current = session.get('current_model')
    
    return jsonify({
        'models': models,
        'current': current
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a single prediction."""
    try:
        current_model = session.get('current_model')
        models = session.get('trained_models', {})
        
        if not current_model or current_model not in models:
            return jsonify({'error': 'No trained model available'}), 400
        
        model_info = models[current_model]
        data = request.get_json()
        
        feature_cols = model_info['feature_cols']
        input_data = {col: float(data.get(col, 0)) for col in feature_cols}
        
        X = pd.DataFrame([input_data])
        
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        
        X = X[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        model_path = model_info['model']
        model_type = model_info['model_type']
        
        if model_type == 'ensemble':
            models_dict = joblib.load(model_path)
            proba = ensemble_predict(models_dict, X)[0]
            prediction = 1 if proba > 0.5 else 0
        else:
            model = joblib.load(model_path)
            prediction = int(model.predict(X)[0])
            proba = float(model.predict_proba(X)[0][1])
        
        return jsonify({
            'prediction': prediction,
            'probability': proba,
            'severity': 'high' if proba > 0.7 else 'medium' if proba > 0.3 else 'low'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Make batch predictions on loaded data."""
    try:
        data_json = session.get('data')
        current_model = session.get('current_model')
        models = session.get('trained_models', {})
        
        if not data_json:
            return jsonify({'error': 'No data loaded'}), 400
        
        if not current_model or current_model not in models:
            return jsonify({'error': 'No trained model available'}), 400
        
        df = pd.read_json(data_json)
        df = df.tail(100)
        
        model_info = models[current_model]
        feature_cols = model_info['feature_cols']
        model_path = model_info['model']
        model_type = model_info['model_type']
        
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        if model_type == 'ensemble':
            models_dict = joblib.load(model_path)
            probas = ensemble_predict(models_dict, X)
            predictions = (probas > 0.5).astype(int)
        else:
            model = joblib.load(model_path)
            predictions = model.predict(X)
            probas = model.predict_proba(X)[:, 1]
        
        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            results.append({
                'timestamp': str(row.get('Timestamp', '')),
                'prediction': 'Failure' if predictions[i] == 1 else 'Normal',
                'probability': float(probas[i])
            })
        
        return jsonify({
            'predictions': results,
            'summary': {
                'total': len(results),
                'failures': int((predictions == 1).sum()),
                'normal': int((predictions == 0).sum())
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)
