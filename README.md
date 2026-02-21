# SecurePay - Bank-Style Transaction Anomaly Monitoring

SecurePay is a Streamlit-based anomaly detection dashboard for financial transactions.  
It uses behavioral features and an Isolation Forest model to flag suspicious patterns and present results in an operations-style monitoring console.

## What This App Does

- Ingests transaction CSV data from upload or demo dataset
- Validates a fixed 8-column schema
- Scores transactions with Isolation Forest
- Generates risk score, severity, and high-risk flags
- Shows executive, operations, model-health, and case views
- Exports scored data, suspicious cases, and executive PDF snapshot

## Fixed Input Schema (Required)

Your dataset must include exactly these columns:

1. `txn_id`
2. `txn_hour`
3. `txn_amount`
4. `amount_deviation`
5. `txn_velocity`
6. `behavior_score`
7. `payment_channel`
8. `risk_flag`

The app is intentionally schema-locked to this structure so datasets with the same columns work consistently.

## Dashboard Structure

- `Executive` tab:
  - Risk score distribution
  - Hourly suspicious trend vs normal baseline
  - Channel anomaly breakdown
  - Amount by severity
- `Risk Operations` tab:
  - Alert funnel
  - Channel-hour risk heatmap
  - Top risky transactions (Pareto)
  - Potential vs prevented loss view
- `Model Health` tab:
  - Confusion matrix
  - Precision-recall curve
  - Drift trend
  - Threshold and contamination status
- `Case Console` tab:
  - High-risk queue
  - Manager/Analyst views
  - Drill-down with reason codes

## Key Metrics Available

- Total transactions and total transaction value
- Suspicious count, suspicious value, high-risk rate
- Blocked suspicious count and estimated prevented loss
- Precision, recall, false positive rate
- Open alerts, open cases
- Average and P95 estimated detection latency
- Data quality score

## Exports

- `securepay_scored_transactions.csv`
- `securepay_suspicious_transactions.csv`
- `securepay_executive_snapshot.pdf` (when matplotlib is available)

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Dependency Notes

- `plotly` is used for interactive charts.
- `matplotlib` is used for PDF snapshot export.
- If one of these is missing, the app does not crash:
  - chart/table fallbacks are shown when `plotly` is unavailable
  - PDF export is disabled when `matplotlib` is unavailable

## Project Files

- `app.py`: Streamlit application
- `requirements.txt`: Python dependencies
- `securepay_txn_demo.csv`: Demo dataset
- `securepay_txn_stream.csv`: Sample stream-like dataset
- notebooks:
  - `01_transaction_landscape_analysis.ipynb`
  - `02_behavioral_feature_preparation.ipynb`
  - `03_isolation_forest_anomaly_detection.ipynb`
  - `04_lof_and_model_evaluation.ipynb`
  - `05_precision_recall_and_threshold.ipynb`

## License

Apache 2.0 (see `LICENSE.txt`).

## Author

Mehul Kumar  
Academic project - Intelligent anomaly detection system.
