# SecurePay — Intelligent Financial Anomaly Detection

SecurePay is a behavioral anomaly detection system designed to identify suspicious financial transaction patterns using unsupervised machine learning. The system analyzes transaction behavior instead of relying on fixed fraud rules, enabling adaptive and data-driven anomaly detection.

This project demonstrates how anomaly detection models can be applied to financial behavioral data and deployed through an interactive web application.

---

## Overview

SecurePay detects unusual transaction behavior using machine learning models such as Isolation Forest and Local Outlier Factor (LOF). The system focuses on behavioral indicators including transaction amount deviation, transaction velocity, and behavioral scoring rather than predefined fraud rules.

The application is built using Streamlit and provides an interactive interface for uploading datasets, running anomaly detection, and visualizing suspicious transactions.

---

## Key Features

* Behavioral anomaly detection using Isolation Forest
* Clean and interactive Streamlit web interface
* Real-time detection on uploaded datasets
* Suspicious transaction highlighting
* Detection summary and anomaly rate
* Visualization of anomaly distribution
* Modular notebook-based workflow
* Fully deployable on Streamlit Cloud

---

## Project Structure

```
SecurePay/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Dependencies for deployment
├── securepay_txn_stream.csv        # Sample dataset
│
├── notebooks/
│   ├── 01_transaction_landscape_analysis.ipynb
│   ├── 02_behavioral_feature_preparation.ipynb
│   ├── 03_isolation_forest_anomaly_detection.ipynb
│   ├── 04_lof_and_model_evaluation.ipynb
│   └── 05_precision_recall_and_threshold.ipynb
│
├── LICENSE
└── README.md
```

---

## Dataset Description

The model expects behavioral transaction data with the following columns:

* **txn_hour** — Hour of transaction (0–23)
* **txn_amount** — Transaction amount
* **amount_deviation** — Deviation from user’s normal spending
* **txn_velocity** — Frequency of transactions in a short time window
* **behavior_score** — Normalized behavioral risk indicator

Synthetic datasets can be generated for testing using tools such as Mockaroo or obtained from open financial datasets.

---

## How It Works

1. Transaction behavioral data is loaded.
2. Behavioral features are extracted.
3. Isolation Forest identifies anomalous patterns.
4. Suspicious transactions are flagged.
5. Detection metrics and visualization are produced.
6. Results are displayed through the Streamlit web interface.

---

## Running the Application

### Local Run

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
streamlit run app.py
```

---

### Streamlit Cloud Deployment

1. Upload project to GitHub
2. Go to https://share.streamlit.io
3. Deploy using:

   * Repository → SecurePay
   * Branch → main
   * File → app.py

The app will run directly in the browser.

---

## Technologies Used

* Python
* Streamlit
* Pandas & NumPy
* Scikit-learn
* Matplotlib

---

## Academic Purpose

This project is developed as an academic demonstration of anomaly detection techniques applied to financial behavioral data. It is intended for learning, experimentation, and research purposes only.

---

## Author

Mehul Kumar
Academic Project — Intelligent Anomaly Detection System

---

## License

This project is licensed under the Apache 2.0 License.

---

## Copyright

© 2026 SecurePay — Intelligent Financial Anomaly Detection
All rights reserved. This project is developed for academic and research demonstration purposes.
