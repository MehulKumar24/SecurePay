import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SecurePay Anomaly Detection",
    layout="wide"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align:center;'>SecurePay — Intelligent Transaction Monitoring</h1>
    <p style='text-align:center; font-size:16px;'>
    Real-time behavioral anomaly detection system for identifying suspicious financial transactions.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------------- FILE UPLOAD ----------------
st.subheader("Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file containing transaction behavioral data",
    type=["csv"]
)

# ---------------- MAIN PROCESS ----------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    required_cols = [
        'txn_hour',
        'txn_amount',
        'amount_deviation',
        'txn_velocity',
        'behavior_score'
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"Dataset missing required columns: {missing}")
        st.stop()

    X = df[required_cols]

    # ---------------- MODEL ----------------
    model = IsolationForest(
        n_estimators=100,
        contamination=0.015,
        random_state=42
    )

    model.fit(X)

    df['anomaly'] = model.predict(X)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

    total_txn = len(df)
    total_anomaly = int(df['anomaly'].sum())
    normal_txn = total_txn - total_anomaly
    anomaly_rate = (total_anomaly / total_txn) * 100

    # ---------------- METRICS ----------------
    st.subheader("Detection Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", total_txn)
    col2.metric("Suspicious Transactions", total_anomaly)
    col3.metric("Normal Transactions", normal_txn)
    col4.metric("Anomaly Rate (%)", f"{anomaly_rate:.2f}")

    st.markdown("---")

    # ---------------- ANOMALY TABLE ----------------
    st.subheader("Detected Suspicious Transactions")

    anomalies = df[df['anomaly'] == 1]

    if len(anomalies) > 0:
        st.dataframe(anomalies.head(50))
    else:
        st.success("No suspicious transactions detected.")

    st.markdown("---")

    # ---------------- GRAPH ----------------
    st.subheader("Anomaly Distribution")

    fig, ax = plt.subplots(figsize=(4,3))
    df['anomaly'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xticklabels(["Normal", "Anomaly"], rotation=0)
    ax.set_ylabel("Count")
    st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:13px;'>
    © 2026 SecurePay | Intelligent Anomaly Detection System  
    Developed for academic research and anomaly detection demonstration.
    </p>
    """,
    unsafe_allow_html=True
)
