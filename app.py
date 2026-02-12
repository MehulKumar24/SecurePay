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

.main-title {
    font-size: 36px;
    font-weight: 700;
    color: #2F80ED;
    text-align: center;
    margin-bottom: 0px;
}

.sub-title {
    font-size: 16px;
    color: #9aa4b2;
    text-align: center;
    margin-bottom: 25px;
}

.metric-box {
    background-color: #111827;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
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