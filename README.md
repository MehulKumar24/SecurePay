# SecurePay — Intelligent Financial Anomaly Detection System

**SecurePay** is an enterprise-grade behavioral anomaly detection system designed to identify suspicious financial transaction patterns using advanced unsupervised machine learning techniques. The system analyzes transaction behavior adaptively, enabling data-driven anomaly detection without relying on fixed fraud rules.

**Status**: ✅ Production Ready | **Version**: 2.0 | **License**: Apache 2.0

---

## 🎯 Overview

SecurePay leverages multiple anomaly detection algorithms (Isolation Forest, Local Outlier Factor) combined with advanced explainability tools (SHAP) and temporal analysis to detect sophisticated fraud patterns. The system is fully deployable through an interactive Streamlit web application with comprehensive compliance and audit logging.

### Core Capabilities

- **Multi-Model Detection**: Isolation Forest + Local Outlier Factor (LOF) ensemble
- **Explainable AI**: SHAP-based feature importance and decision explanation
- **Temporal Analysis**: Hourly patterns, daily trends, and temporal anomaly detection
- **Risk Scoring**: Multi-level risk classification (Low, Medium, High)
- **Audit Compliance**: Complete audit logging and GDPR-compliant reporting
- **Data Quality**: Comprehensive data validation and quality assessment
- **Advanced Visualization**: Interactive dashboards with Plotly & Matplotlib
- **Export Capabilities**: HTML, JSON, CSV, and compliance reports

---

## ✨ Key Features

✅ **Advanced Detection**
- Dual-algorithm ensemble for improved accuracy
- Confidence scoring for predictions
- Anomaly rate trending and temporal pattern detection

✅ **Enterprise Features**
- Comprehensive audit logging (AuditLogger)
- Data quality validation (DataQualityValidator)
- Compliance and GDPR-ready reports (ReportGenerator)
- Feature importance analysis (ShapAnalyzer)

✅ **Analytics & Insights**
- Performance metrics calculation
- ROC curves and precision-recall analysis
- Hourly and daily pattern analysis
- Feature contribution analysis

✅ **User Experience**
- Interactive Streamlit web interface
- Real-time detection on uploaded datasets
- Advanced filtering and risk analysis
- Payment channel risk analysis

✅ **Developer Features**
- Modular utility architecture
- Fully type-hinted code
- Comprehensive error handling
- Production-ready deployment

---

## 📂 Project Structure

```
SecurePay/
├── app.py                              # Main Streamlit web application
├── app_original.py                     # Original app reference
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
├── LICENSE                             # Apache 2.0 License
│
├── .streamlit/
│   └── config.toml                     # Streamlit configuration
│
├── .vscode/
│   └── settings.json                   # VS Code Python interpreter settings
│
├── utils/                              # Core utility modules
│   ├── __init__.py                     # Package initialization
│   ├── audit_logger.py                 # Audit logging & compliance tracking
│   ├── data_quality.py                 # Data validation & quality assessment
│   ├── metrics.py                      # Model performance metrics calculator
│   ├── reporting.py                    # Report generation (HTML, JSON, CSV)
│   ├── shap_analysis.py                # SHAP-based explainability analysis
│   └── time_series_analysis.py         # Temporal pattern analysis
│
├── notebooks/
│   ├── 01_transaction_landscape_analysis.ipynb
│   ├── 02_behavioral_feature_preparation.ipynb
│   ├── 03_isolation_forest_anomaly_detection.ipynb
│   ├── 04_lof_and_model_evaluation.ipynb
│   └── 05_precision_recall_and_threshold.ipynb
│
├── data/
│   ├── securepay_txn_demo.csv          # Demo dataset (10K records)
│   └── securepay_txn_stream.csv        # Stream dataset for real-time testing
│
└── generated_reports/                  # Output directory for generated reports
```

---

## 📊 Dataset & Features

### Required Features

The model expects behavioral transaction data with these columns:

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `txn_hour` | Integer | 0-23 | Transaction hour of day |
| `txn_amount` | Float | Variable | Transaction amount in currency units |
| `amount_deviation` | Float | 0-1 | Normalized deviation from user's typical spending |
| `txn_velocity` | Float | 0-100 | Transaction frequency (transactions per time window) |
| `behavior_score` | Float | 0-1 | Aggregated behavioral risk indicator |

### Sample Datasets

Two sample datasets are included:
- **`securepay_txn_demo.csv`**: 10,000 sample transactions (production testing)
- **`securepay_txn_stream.csv`**: Real-time transaction stream format

### Data Quality Validation

The system automatically validates:
- ✅ Missing values detection and reporting
- ✅ Data type consistency checking
- ✅ Duplicate record identification
- ✅ Outlier detection (IQR & Z-score methods)
- ✅ Memory usage profiling
- ✅ Comprehensive quality scoring (0-100)

---

## 🔄 How It Works

### Detection Pipeline

```
Input CSV → Validation → Feature Check → Model Training → Detection
    ↓          ↓            ↓              ↓                ↓
  Upload   Data Quality  Required      Isolation    Risk Scoring
          Assessment     Features      Forest + LOF
                                                        ↓
                                          Risk Classification
                                          & Visualization
```

### Step-by-Step Workflow

1. **Data Upload** - User uploads CSV with transaction data
2. **Validation** - Automatic data quality check and feature validation
3. **Feature Engineering** - Ensures all required features are present
4. **Model Training** - Trains both Isolation Forest and LOF models
5. **Prediction** - Generates anomaly scores and predictions
6. **Risk Classification** - Assigns risk levels (Low, Medium, High)
7. **Analysis** - Temporal patterns, feature importance, audit logging
8. **Visualization** - Interactive charts and detailed reports
9. **Export** - HTML, JSON, CSV, and compliance reports

### Anomaly Detection Models

**Isolation Forest**
- Excellent for high-dimensional data
- Fast and scalable
- Contamination parameter: 1.5%
- Estimators: 100 trees

**Local Outlier Factor (LOF)**
- Density-based approach
- Captures local outliers
- K-neighbors: 20
- Contamination parameter: 1.5%

### Ensemble Scoring

Both models are combined for robust predictions:
- Voting consensus approach
- Individual scores reported
- Confidence intervals calculated

---

## 🚀 Running the Application

### Prerequisites

- Python 3.9+
- Git
- Pip or Conda

### Local Installation & Setup

1. **Clone the Repository**
```bash
git clone https://github.com/MehulKumar24/SecurePay.git
cd SecurePay
```

2. **Create Python Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure VS Code (Optional)**
Open Command Palette (`Ctrl+Shift+P`) and select:
- `Python: Select Interpreter`
- Choose `./.venv/bin/python`

### Running the Streamlit Application

```bash
streamlit run app.py
```

The app will launch at **`http://localhost:8501`**

### Application Features

**Main Dashboard**
- 📤 CSV file upload (max 500MB)
- 🎯 Real-time anomaly detection
- 📊 Interactive visualizations
- 📈 Performance metrics display

**Advanced Analytics**
- ⏰ Temporal pattern analysis (hourly/daily)
- 💳 Payment channel risk profiling
- 🔍 Feature importance via SHAP
- 📋 Comprehensive audit logging

**Reports & Exports**
- 📄 HTML professional reports
- 📑 JSON machine-readable format
- 📊 CSV anomaly lists
- ✅ GDPR compliance reports

---

### Streamlit Cloud Deployment

1. **Push to GitHub**
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Click "New app"
   - Connect your GitHub repository
   - Select repository, branch (main), and file (app.py)
   - Deploy!

   The app will run directly in the browser.

---

## 🛠️ Technologies & Dependencies

### Core Framework
- **Streamlit 1.28+** - Interactive web framework
- **Python 3.11+** - Programming language

### Data & ML Libraries
- **Pandas 2.0+** - Data manipulation
- **NumPy 1.23+** - Numerical computing
- **Scikit-learn 1.3+** - Machine learning algorithms
  - Isolation Forest
  - Local Outlier Factor
  - Model metrics and evaluation

### Visualization
- **Matplotlib 3.7+** - Core plotting
- **Seaborn 0.12+** - Statistical visualization
- **Plotly 5.13+** - Interactive charts

### Explainability & Analysis
- **SHAP 0.42+** - Feature importance analysis
- **Apex Charts (via Plotly)** - Advanced visualizations

### Support Libraries
- **ReportLab 4.0+** - PDF generation
- **Jinja2 3.1+** - HTML templating
- **SQLAlchemy 2.0+** - Database abstraction
- **Pydantic 2.0+** - Data validation
- **Cryptography 41.0+** - Security utilities
- **Python-dotenv 1.0+** - Environment management

### Development Tools
- **Git** - Version control
- **Pip** - Package management
- **Virtual Environment** - Dependency isolation

See `requirements.txt` for complete dependency list with versions.

---

## 📚 API & Utility Modules Reference

### Core Utility Modules

#### `metrics.ModelMetricsCalculator`
```python
# Calculate comprehensive model metrics
metrics = ModelMetricsCalculator.calculate_metrics(
    y_true=labels,
    y_pred=predictions,
    y_scores=probabilities
)
# Returns: confusion matrix, precision, recall, F1, ROC-AUC, PR curves

# Confidence intervals
conf = ModelMetricsCalculator.calculate_confidence_intervals(predictions)

# Feature statistics
stats = ModelMetricsCalculator.get_feature_statistics(X_df, features)
```

#### `audit_logger.AuditLogger`
```python
# Log system actions for compliance
AuditLogger.log_action(
    action_type='data_upload',
    details={'filename': 'transactions.csv', 'rows': 10000},
    user_id='analyst_001'
)

# Retrieve audit history
history = AuditLogger.get_audit_history(days=30)

# Generate compliance report
report = AuditLogger.get_compliance_report()
```

#### `data_quality.DataQualityValidator`
```python
# Assess data quality
assessment = DataQualityValidator.assess_data_quality(df)

# Validate required features
validation = DataQualityValidator.validate_features(df, required_features)

# Detect outliers
outliers = DataQualityValidator.detect_outliers(
    df, 
    features=['txn_amount', 'txn_velocity'],
    method='iqr',
    threshold=1.5
)

# Get comprehensive profile
profile = DataQualityValidator.get_data_profile(df)
```

#### `shap_analysis.ShapAnalyzer`
```python
# Calculate feature importance
importance = ShapAnalyzer.calculate_feature_importance(X, model, features)

# Explain individual predictions
explanation = ShapAnalyzer.explain_prediction(
    sample=transaction_record,
    model=model,
    features=feature_list,
    X_reference=reference_dataset
)

# Get anomaly profile
profile = ShapAnalyzer.get_anomaly_profile(df)
```

#### `time_series_analysis.TimeSeriesAnalyzer`
```python
# Analyze hourly patterns
hourly = TimeSeriesAnalyzer.analyze_hourly_patterns(df)

# Daily pattern analysis
daily = TimeSeriesAnalyzer.analyze_daily_patterns(df)

# Detect trends
trends = TimeSeriesAnalyzer.detect_temporal_trends(df)

# Get peak anomaly times
peaks = TimeSeriesAnalyzer.get_peak_anomaly_times(df)

# Temporal statistics
stats = TimeSeriesAnalyzer.temporal_statistics(df)
```

#### `reporting.ReportGenerator`
```python
# Generate HTML report
html_file = ReportGenerator.generate_html_report(
    df, anomalies, metrics, data_quality
)

# Generate JSON report
json_report = ReportGenerator.generate_json_report(
    df, anomalies, metrics, data_quality
)

# Generate compliance report
compliance = ReportGenerator.generate_compliance_report(
    df, audit_logs, data_quality
)

# Export anomalies to CSV
csv_file = ReportGenerator.generate_csv_export(df, anomalies)

# List all reports
reports = ReportGenerator.list_reports()
```

---

## 💡 Usage Examples

### Basic Detection
```python
import pandas as pd
from sklearn.ensemble import IsolationForest

# Load data
df = pd.read_csv('transactions.csv')

# Train model
model = IsolationForest(contamination=0.015, random_state=42)
df['anomaly'] = model.fit_predict(df[features])

# Get results
anomalies = df[df['anomaly'] == -1]
print(f"Detected {len(anomalies)} anomalies")
```

### With Risk Scoring
```python
from utils.shap_analysis import ShapAnalyzer

# Get feature importance
importance = ShapAnalyzer.calculate_feature_importance(X, model, features)

# Classify risk levels
df['risk_level'] = pd.cut(
    df['anomaly_score'],
    bins=[0, 0.33, 0.67, 1.0],
    labels=['Low', 'Medium', 'High']
)
```

### Generate Reports
```python
from utils.reporting import ReportGenerator
from utils.data_quality import DataQualityValidator

# Validate data quality
quality = DataQualityValidator.assess_data_quality(df)

# Generate report
ReportGenerator.generate_html_report(
    df, 
    anomalies, 
    metrics, 
    quality
)
```

---

## 📖 Academic Purpose

SecurePay is developed as an **academic research project** demonstrating:
- Advanced anomaly detection techniques
- Behavioral financial analysis
- Machine learning in fraud detection
- Enterprise-grade system design
- Compliance and audit requirements

This project is suitable for:
- ✅ Educational purposes
- ✅ Research demonstrations
- ✅ Proof-of-concepts
- ✅ Capstone projects
- ✅ Portfolio development

---

## 👨‍💻 Author & Contributing

**Primary Developer**: Mehul Kumar  
**Project Status**: Active Development  
**Last Updated**: February 2026

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### Support & Issues

For issues, bugs, or feature requests:
- Open an issue on GitHub
- Include reproduction steps
- Provide sample data if possible

---

## 📄 License

This project is licensed under the **Apache License 2.0**

```
Copyright © 2026 SecurePay Contributors
Licensed under the Apache License, Version 2.0
```

You are free to:
- ✅ Use commercially
- ✅ Modify the code
- ✅ Distribute copies
- ✅ Use privately

With conditions:
- License and copyright notice must be included
- Changes must be documented

See [LICENSE](LICENSE) file for complete terms.

---

## ⚖️ Copyright & Usage

© 2026 **SecurePay** — Intelligent Financial Anomaly Detection System  
All rights reserved.

This project is developed for **academic and research demonstration purposes**. While it demonstrates real-world techniques, it should not be used in production environments without substantial security auditing and regulatory compliance review.

### Important Disclaimers

⚠️ **Not for Production Use** - This is a research prototype  
⚠️ **Educational Purpose** - Intended for learning and demonstration  
⚠️ **Research Grade** - Requires additional hardening for financial systems  
⚠️ **No Warranty** - Provided as-is without guarantees  

### Regulatory Compliance

Users are responsible for:
- Compliance with financial regulations (PCI-DSS, etc.)
- Data protection laws (GDPR, CCPA, etc.)
- Security and encryption standards
- User consent and privacy policies

---

## 📞 Contact & Support

**Project Repository**: [github.com/MehulKumar24/SecurePay](https://github.com/MehulKumar24/SecurePay)

**Questions?** Open an issue or discussion on GitHub.

---

## 🙏 Acknowledgments

- Scikit-learn team for exceptional ML algorithms
- Streamlit for interactive web framework
- SHAP team for explainability research
- The open-source Python community

---

**Happy Detecting! 🔍**
