import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SecurePay — Suspicious Transaction Detection",
    page_icon="💳",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>

/* -------- GLOBAL -------- */
html, body, [class*="css"]  {
    background-color: #0b0f1a;
    color: #e6edf3;
    font-family: "Segoe UI", Roboto, sans-serif;
}

/* -------- MAIN TITLE -------- */
.title {
    font-size: 38px;
    font-weight: 600;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 4px;
}

.blue {
    color: #3b82f6;
}

.subtitle {
    text-align: center;
    color: #9aa4b2;
    font-size: 15px;
    margin-bottom: 28px;
}

/* -------- FILE UPLOADER (CENTER CARD STYLE) -------- */
[data-testid="stFileUploader"] {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 30px;
    width: 60%;
    margin: auto;
    box-shadow: 0 0 0 1px rgba(59,130,246,0.05);
}

/* Upload text */
[data-testid="stFileUploader"] label {
    color: #cbd5e1;
    font-size: 16px;
}

/* Browse button */
button[kind="secondary"] {
    background-color: #1f2937 !important;
    color: #e5e7eb !important;
    border-radius: 8px !important;
    border: 1px solid #374151 !important;
}

button[kind="secondary"]:hover {
    border: 1px solid #3b82f6 !important;
    color: #3b82f6 !important;
}

/* -------- METRICS -------- */
[data-testid="stMetric"] {
    background: #111827;
    border-radius: 10px;
    padding: 14px;
    border: 1px solid #1f2937;
}

/* -------- TABLE -------- */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 1px solid #1f2937;
}

/* -------- FOOTER -------- */
.footer {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    margin-top: 18px;
}

/* -------- SCROLLBAR -------- */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #1f2937;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("SecurePay")
    st.write("Intelligent Financial Anomaly Detection System")

    st.markdown("---")
    st.markdown("### About")
    st.write(
        "SecurePay detects suspicious financial transactions using behavioural anomaly detection techniques."
    )

    st.markdown("### Features")
    st.write("• Real-time anomaly detection")
    st.write("• Behaviour-based risk scoring")
    st.write("• Visual analytics dashboard")
    st.write("• Works with large datasets")

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">SecurePay</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Intelligent Financial Anomaly Detection</p>', unsafe_allow_html=True)

st.write("Upload dataset to analyse behavioural transaction anomalies")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ---------------- MODEL ----------------
    features = df.select_dtypes(include=[np.number]).fillna(0)

    model = IsolationForest(
        n_estimators=150,
        contamination="auto",
        random_state=42
    )

    model.fit(features)
    preds = model.predict(features)

    df["anomaly_flag"] = np.where(preds == -1, 1, 0)

    # ---------------- METRICS ----------------
    total_txn = len(df)
    suspicious = df["anomaly_flag"].sum()
    anomaly_rate = (suspicious / total_txn) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Transactions", f"{total_txn:,}")

    with col2:
        st.metric("Suspicious Detected", f"{suspicious:,}")

    with col3:
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

    st.markdown("---")

    # ---------------- DATA PREVIEW ----------------
    st.subheader("Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True)

    # ---------------- SUSPICIOUS ONLY ----------------
    st.subheader("Detected Suspicious Transactions")
    suspicious_df = df[df["anomaly_flag"] == 1]
    st.dataframe(suspicious_df.head(50), use_container_width=True)

    # ---------------- VISUAL ----------------
    st.subheader("Anomaly Distribution")

    fig = px.histogram(
        df,
        x="anomaly_flag",
        nbins=2,
        title="Normal vs Suspicious Transactions"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr style="border:0.5px solid #1f2937">
    <center>
    © 2026 SecurePay — Intelligent Financial Anomaly Detection System<br>
    Academic Research Prototype • Developed by Mehul Kumar
    </center>
    """,
    unsafe_allow_html=True
)