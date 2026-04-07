const API_BASE = '';

function showLoading(text = 'Loading...') {
    const overlay = document.getElementById('loading');
    if (overlay) {
        overlay.querySelector('p').textContent = text;
        overlay.classList.add('active');
    }
}

function hideLoading() {
    const overlay = document.getElementById('loading');
    if (overlay) {
        overlay.classList.remove('active');
    }
}

function showError(message) {
    alert('Error: ' + message);
}

function showSuccess(message) {
    alert('Success: ' + message);
}

async function apiCall(endpoint, method = 'GET', data = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json'
        }
    };
    
    if (data) {
        options.body = JSON.stringify(data);
    }
    
    try {
        const response = await fetch(API_BASE + endpoint, options);
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'API call failed');
        }
        
        return result;
    } catch (error) {
        throw error;
    }
}

async function loadDashboardStats() {
    try {
        const info = await apiCall('/api/get-data-info');
        const models = await apiCall('/api/get-trained-models');
        
        document.getElementById('stat-records').textContent = info.loaded ? info.rows.toLocaleString() : '0';
        document.getElementById('stat-models').textContent = Object.keys(models.models || {}).length;
        
        if (info.loaded) {
            const df = await apiCall('/api/get-data-preview');
            if (df.rows && df.rows.length > 0) {
                const sample = df.rows.slice(0, 100);
                const powers = sample.map(r => r.Power).filter(p => p !== undefined);
                const avgPower = powers.length > 0 ? (powers.reduce((a, b) => a + b, 0) / powers.length).toFixed(2) : '0';
                document.getElementById('stat-power').textContent = avgPower + ' kW';
                
                const dutyCycle = powers.filter(p => p > 0).length / powers.length * 100;
                document.getElementById('stat-duty').textContent = dutyCycle.toFixed(1) + '%';
            }
        }
    } catch (error) {
        console.error('Failed to load dashboard stats:', error);
    }
}

async function loadSampleData() {
    showLoading('Loading sample data...');
    try {
        const result = await apiCall('/api/load-sample-data');
        hideLoading();
        showSuccess('Loaded ' + result.rows + ' records');
        checkDataStatus();
        loadDashboardStats();
    } catch (error) {
        hideLoading();
        showError(error.message);
    }
}

async function uploadFile() {
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];
    
    if (!file) {
        showError('Please select a file');
        return;
    }
    
    showLoading('Uploading file...');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(API_BASE + '/api/upload-data', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        hideLoading();
        
        if (!response.ok) {
            throw new Error(result.error || 'Upload failed');
        }
        
        showSuccess('Uploaded ' + result.rows + ' records');
        checkDataStatus();
        loadDashboardStats();
    } catch (error) {
        hideLoading();
        showError(error.message);
    }
}

async function checkDataStatus() {
    const statusEl = document.getElementById('data-status');
    if (!statusEl) return;
    
    try {
        const info = await apiCall('/api/get-data-info');
        
        if (!info.loaded) {
            statusEl.innerHTML = `
                <div class="empty-state">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"></path></svg>
                    <h3>No Data Loaded</h3>
                    <p>Upload a CSV file or load sample data to get started</p>
                </div>
            `;
            return;
        }
        
        statusEl.innerHTML = `
            <div class="grid grid-3">
                <div class="stat-card">
                    <div class="stat-label">Total Records</div>
                    <div class="stat-value">${info.rows.toLocaleString()}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Columns</div>
                    <div class="stat-value">${info.columns.length}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Missing Values</div>
                    <div class="stat-value">${info.missing_values.toLocaleString()}</div>
                </div>
            </div>
        `;
        
        loadDataPreview();
    } catch (error) {
        statusEl.innerHTML = `
            <div class="alert alert-danger">
                <p>Failed to check data status: ${error.message}</p>
            </div>
        `;
    }
}

async function loadDataPreview() {
    const previewBody = document.getElementById('data-preview-body');
    if (!previewBody) return;
    
    try {
        const data = await apiCall('/api/get-data-preview');
        
        if (!data.rows || data.rows.length === 0) {
            previewBody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-secondary);">No data available</td></tr>';
            return;
        }
        
        const sample = data.rows.slice(0, 50);
        const columns = data.columns.slice(0, 6);
        
        let html = '';
        for (const row of sample) {
            html += '<tr>';
            for (const col of columns) {
                let val = row[col];
                if (val === undefined || val === null) val = '';
                if (typeof val === 'number') val = val.toFixed(2);
                html += `<td>${val}</td>`;
            }
            html += '</tr>';
        }
        
        const headerHtml = columns.map(col => `<th>${col}</th>`).join('');
        document.querySelector('#data-preview-table thead tr').innerHTML = headerHtml;
        previewBody.innerHTML = html;
        
    } catch (error) {
        previewBody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--danger-color);">Failed to load preview</td></tr>';
    }
}

function setupSliders() {
    const sliders = [
        { id: 'n-estimators', valId: 'n-est-val', suffix: '' },
        { id: 'max-depth', valId: 'max-d-val', suffix: '' },
        { id: 'test-size', valId: 'test-s-val', suffix: '%', transform: v => v },
        { id: 'failure-threshold', valId: 'thresh-val', suffix: '', transform: v => (v / 10).toFixed(1) }
    ];
    
    for (const slider of sliders) {
        const el = document.getElementById(slider.id);
        const valEl = document.getElementById(slider.valId);
        
        if (el && valEl) {
            el.addEventListener('input', function() {
                const val = slider.transform ? slider.transform(this.value) : this.value;
                valEl.textContent = val + slider.suffix;
            });
        }
    }
}

async function checkDataRequired() {
    try {
        const info = await apiCall('/api/get-data-info');
        if (!info.loaded) {
            const resultsEl = document.getElementById('training-results');
            if (resultsEl) {
                resultsEl.innerHTML = `
                    <div class="alert alert-warning">
                        <p>Please load data first before training. Go to the Data page.</p>
                    </div>
                `;
            }
        }
    } catch (error) {
        console.error('Failed to check data status:', error);
    }
}

async function loadTrainedModels() {
    const listEl = document.getElementById('trained-models-list');
    if (!listEl) return;
    
    try {
        const data = await apiCall('/api/get-trained-models');
        const models = data.models || {};
        const modelKeys = Object.keys(models);
        
        if (modelKeys.length === 0) {
            listEl.innerHTML = `
                <div class="empty-state">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path></svg>
                    <h3>No Models Trained</h3>
                    <p>Train your first model to get started</p>
                </div>
            `;
            return;
        }
        
        let html = '';
        for (const name of modelKeys) {
            const info = models[name];
            const metrics = info.metrics || {};
            const isCurrent = data.current === name;
            
            html += `
                <div class="model-card">
                    <div class="model-info">
                        <h4>${name} ${isCurrent ? '<span class="badge badge-primary">Active</span>' : ''}</h4>
                        <p>Type: ${info.model_type} | Accuracy: ${((metrics.accuracy || 0) * 100).toFixed(1)}%</p>
                    </div>
                    <div class="model-actions">
                        ${!isCurrent ? '<button class="btn btn-sm btn-secondary">Use</button>' : ''}
                    </div>
                </div>
            `;
        }
        
        listEl.innerHTML = html;
        
    } catch (error) {
        listEl.innerHTML = `
            <div class="alert alert-danger">
                <p>Failed to load models: ${error.message}</p>
            </div>
        `;
    }
}

async function trainModel() {
    showLoading('Training model...');
    
    try {
        const info = await apiCall('/api/get-data-info');
        if (!info.loaded) {
            hideLoading();
            showError('Please load data first');
            return;
        }
        
        const config = {
            model_type: document.getElementById('model-type').value,
            n_estimators: parseInt(document.getElementById('n-estimators').value),
            max_depth: parseInt(document.getElementById('max-depth').value),
            test_size: parseInt(document.getElementById('test-size').value) / 100,
            failure_threshold: parseFloat(document.getElementById('failure-threshold').value) / 10,
            balance_method: document.getElementById('balance-method').value
        };
        
        const result = await apiCall('/api/train-model', 'POST', config);
        hideLoading();
        
        if (result.success) {
            showSuccess(`Model trained: ${result.model_name}`);
            displayTrainingResults(result);
            loadTrainedModels();
            loadDashboardStats();
        }
        
    } catch (error) {
        hideLoading();
        showError(error.message);
    }
}

function displayTrainingResults(result) {
    const resultsEl = document.getElementById('training-results');
    if (!resultsEl) return;
    
    const metrics = result.metrics || {};
    
        resultsEl.innerHTML = `
        <div class="alert alert-success" style="margin-bottom: 16px;">
            <p><strong>Model Trained Successfully!</strong></p>
            <p>${result.model_name}</p>
        </div>
        <div class="grid grid-2" style="margin-bottom: 16px;">
            <div class="stat-card">
                <div class="stat-label">Accuracy</div>
                <div class="stat-value">${((metrics.accuracy || 0) * 100).toFixed(1)}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">ROC AUC</div>
                <div class="stat-value">${(metrics.roc_auc || 0).toFixed(2)}</div>
            </div>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: ${(metrics.accuracy || 0) * 100}%;"></div>
        </div>
    `;
}

async function checkModelAvailable() {
    try {
        const data = await apiCall('/api/get-trained-models');
        if (!data.current) {
            const resultEl = document.getElementById('prediction-result');
            if (resultEl) {
                resultEl.innerHTML = `
                    <div class="alert alert-warning">
                        <p>No trained model available. Please train a model first.</p>
                    </div>
                `;
            }
        }
    } catch (error) {
        console.error('Failed to check model:', error);
    }
}

async function makePrediction() {
    showLoading('Making prediction...');
    
    try {
        const features = {
            T_Supply: parseFloat(document.getElementById('T_Supply').value),
            T_Return: parseFloat(document.getElementById('T_Return').value),
            SP_Return: parseFloat(document.getElementById('SP_Return').value),
            T_Saturation: parseFloat(document.getElementById('T_Saturation').value),
            T_Outdoor: parseFloat(document.getElementById('T_Outdoor').value),
            RH_Supply: parseFloat(document.getElementById('RH_Supply').value),
            RH_Return: parseFloat(document.getElementById('RH_Return').value),
            RH_Outdoor: parseFloat(document.getElementById('RH_Outdoor').value),
            Energy: parseFloat(document.getElementById('Energy').value),
            Power: parseFloat(document.getElementById('Power').value)
        };
        
        const result = await apiCall('/api/predict', 'POST', features);
        hideLoading();
        displayPredictionResult(result);
        
    } catch (error) {
        hideLoading();
        showError(error.message);
    }
}

function displayPredictionResult(result) {
    const resultEl = document.getElementById('prediction-result');
    if (!resultEl) return;
    
    const isFailure = result.prediction === 1;
    const severity = result.severity || 'low';
    
    const severityColors = {
        high: 'var(--danger-color)',
        medium: 'var(--warning-color)',
        low: 'var(--success-color)'
    };
    
    resultEl.innerHTML = `
        <div class="grid grid-2" style="margin-bottom: 24px;">
            <div style="text-align: center; padding: 24px; background: ${isFailure ? 'rgba(239, 68, 68, 0.1)' : 'rgba(34, 197, 94, 0.1)'}; border-radius: 12px;">
                <div style="font-size: 48px; margin-bottom: 8px;">${isFailure ? '⚠️' : '✅'}</div>
                <div style="font-size: 20px; font-weight: 600;">${isFailure ? 'FAILURE PREDICTED' : 'NORMAL OPERATION'}</div>
            </div>
            <div style="text-align: center; padding: 24px;">
                <div class="stat-label">Failure Probability</div>
                <div class="stat-value" style="font-size: 36px; color: ${severityColors[severity]};">${(result.probability * 100).toFixed(1)}%</div>
                <div style="margin-top: 8px;">Severity: <strong style="color: ${severityColors[severity]};">${severity.toUpperCase()}</strong></div>
            </div>
        </div>
        ${isFailure ? `
            <div class="alert alert-danger">
                <strong>Recommendations:</strong>
                <ul style="margin: 8px 0 0 20px;">
                    <li>Schedule immediate maintenance inspection</li>
                    <li>Check compressor operation and refrigerant levels</li>
                    <li>Verify heat exchange performance</li>
                    <li>Inspect electrical connections</li>
                </ul>
            </div>
        ` : `
            <div class="alert alert-success">
                <p>Equipment operating normally. Continue monitoring.</p>
            </div>
        `}
    `;
}

async function runBatchPrediction() {
    showLoading('Running batch predictions...');
    
    try {
        const result = await apiCall('/api/batch-predict', 'POST');
        hideLoading();
        displayBatchResults(result);
        
    } catch (error) {
        hideLoading();
        showError(error.message);
    }
}

function displayBatchResults(result) {
    const resultsEl = document.getElementById('batch-results');
    if (!resultsEl) return;
    
    const summary = result.summary || {};
    const predictions = result.predictions || [];
    
    resultsEl.innerHTML = `
        <div class="grid grid-3" style="margin-bottom: 24px;">
            <div class="stat-card">
                <div class="stat-label">Total Predictions</div>
                <div class="stat-value">${summary.total || 0}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Failures Detected</div>
                <div class="stat-value" style="color: var(--danger-color);">${summary.failures || 0}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Normal Operations</div>
                <div class="stat-value" style="color: var(--success-color);">${summary.normal || 0}</div>
            </div>
        </div>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Prediction</th>
                    <th>Probability</th>
                </tr>
            </thead>
            <tbody>
                ${predictions.slice(0, 50).map(p => `
                    <tr>
                        <td>${p.timestamp ? p.timestamp.substring(0, 19) : 'N/A'}</td>
                        <td><span class="badge ${p.prediction === 'Failure' ? 'badge-danger' : 'badge-success'}">${p.prediction}</span></td>
                        <td>${(p.probability * 100).toFixed(1)}%</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

async function loadAnalyticsData() {
    try {
        const info = await apiCall('/api/get-data-info');
        if (!info.loaded) return;
        
        const data = await apiCall('/api/get-data-preview');
        if (!data.rows || data.rows.length === 0) return;
        
        const rows = data.rows;
        
        document.getElementById('a-stat-records').textContent = rows.length.toLocaleString();
        
        const timestamps = rows.map(r => r.Timestamp).filter(t => t);
        if (timestamps.length > 0) {
            const dateStr = timestamps[0].substring(0, 10) + ' ... ' + timestamps[timestamps.length - 1].substring(0, 10);
            document.getElementById('a-stat-date').textContent = dateStr;
        }
        
        const powers = rows.map(r => r.Power).filter(p => p !== undefined);
        if (powers.length > 0) {
            const avgPower = (powers.reduce((a, b) => a + b, 0) / powers.length).toFixed(2);
            document.getElementById('a-stat-power').textContent = avgPower + ' kW';
            
            const dutyCycle = (powers.filter(p => p > 0).length / powers.length * 100).toFixed(1);
            document.getElementById('a-stat-duty').textContent = dutyCycle + '%';
            
            renderPowerChart(rows);
            renderHourlyChart(rows);
            renderDailyChart(rows);
        }
        
    } catch (error) {
        console.error('Failed to load analytics:', error);
    }
}

function renderPowerChart(rows) {
    const chartEl = document.getElementById('chart-power');
    if (!chartEl) return;
    
    const sample = rows.filter((_, i) => i % 10 === 0);
    
    let traces = [{
        x: sample.map(r => r.Timestamp),
        y: sample.map(r => r.Power),
        type: 'scatter',
        mode: 'lines',
        line: { color: '#3b82f6', width: 1 }
    }];
    
    Plotly.newPlot(chartEl, traces, {
        margin: { t: 20, r: 20, b: 40, l: 50 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#94a3b8' },
        xaxis: { 
            gridcolor: '#334155',
            showgrid: true,
            tickformat: '%H:%M'
        },
        yaxis: { 
            gridcolor: '#334155',
            showgrid: true,
            title: 'Power (kW)'
        }
    }, { responsive: true });
}

function renderHourlyChart(rows) {
    const chartEl = document.getElementById('chart-hourly');
    if (!chartEl) return;
    
    const hourlyData = {};
    rows.forEach(r => {
        if (r.Timestamp) {
            const hour = new Date(r.Timestamp).getHours();
            if (!hourlyData[hour]) hourlyData[hour] = [];
            if (r.Power !== undefined) hourlyData[hour].push(r.Power);
        }
    });
    
    const hours = Object.keys(hourlyData).sort();
    const avgs = hours.map(h => {
        const vals = hourlyData[h];
        return vals.reduce((a, b) => a + b, 0) / vals.length;
    });
    
    const traces = [{
        x: hours,
        y: avgs,
        type: 'bar',
        marker: { color: '#3b82f6' }
    }];
    
    Plotly.newPlot(chartEl, traces, {
        margin: { t: 20, r: 20, b: 40, l: 50 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#94a3b8' },
        xaxis: { 
            gridcolor: '#334155',
            title: 'Hour'
        },
        yaxis: { 
            gridcolor: '#334155',
            title: 'Avg Power (kW)'
        }
    }, { responsive: true });
}

function renderDailyChart(rows) {
    const chartEl = document.getElementById('chart-daily');
    if (!chartEl) return;
    
    const dailyData = {};
    const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    
    rows.forEach(r => {
        if (r.Timestamp) {
            const day = new Date(r.Timestamp).getDay();
            if (!dailyData[day]) dailyData[day] = [];
            if (r.Power !== undefined) dailyData[day].push(r.Power);
        }
    });
    
    const days = Object.keys(dailyData).sort();
    const avgs = days.map(d => {
        const vals = dailyData[d];
        return vals.reduce((a, b) => a + b, 0) / vals.length;
    });
    
    const traces = [{
        x: days.map(d => dayNames[parseInt(d)]),
        y: avgs,
        type: 'bar',
        marker: { color: '#22c55e' }
    }];
    
    Plotly.newPlot(chartEl, traces, {
        margin: { t: 20, r: 20, b: 40, l: 50 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#94a3b8' },
        xaxis: { 
            gridcolor: '#334155',
            title: 'Day of Week'
        },
        yaxis: { 
            gridcolor: '#334155',
            title: 'Avg Power (kW)'
        }
    }, { responsive: true });
}

document.addEventListener('DOMContentLoaded', function() {
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-item').forEach(item => {
        if (item.getAttribute('href') === currentPath) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
});
