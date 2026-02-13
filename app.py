import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SecurePay",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.big-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #4da6ff;
}
.subtitle {
    text-align: center;
    color: #b0b3b8;
    margin-bottom: 25px;
}
.metric-card {
    background: #161b22;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0px 0px 8px rgba(0,0,0,0.3);
}
.footer {
    text-align: center;
    color: #8b949e;
    margin-top: 40px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='big-title'>SecurePay</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Intelligent Financial Anomaly Detection System</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- FILE UPLOAD ----------------
st.subheader("Upload Transaction Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# ---------------- MAIN ----------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    features = [
        'txn_hour',
        'txn_amount',
        'amount_deviation',
        'txn_velocity',
        'behavior_score'
    ]

    if not all(col in df.columns for col in features):
        st.error("Dataset missing required columns.")
        st.stop()

    X = df[features]

    model = IsolationForest(
        n_estimators=100,
        contamination=0.015,
        random_state=42
    )

    model.fit(X)

    df['anomaly'] = model.predict(X)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

    total = len(df)
    anomalies = int(df['anomaly'].sum())
    normal = total - anomalies
    rate = (anomalies / total) * 100

    st.subheader("Detection Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", total)
    c2.metric("Suspicious", anomalies)
    c3.metric("Normal", normal)
    c4.metric("Anomaly Rate (%)", f"{rate:.2f}")

    st.markdown("---")

    st.subheader("Detected Suspicious Transactions")

    flagged = df[df['anomaly'] == 1]

    if len(flagged) > 0:
        st.dataframe(flagged.head(50), use_container_width=True)
    else:
        st.success("No suspicious transactions detected.")

    st.markdown("---")

    st.subheader("Anomaly Distribution")

    fig, ax = plt.subplots()
    df['anomaly'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xticklabels(["Normal", "Anomaly"], rotation=0)
    st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("""
<div class='footer'>
© 2026 SecurePay | Intelligent Transaction Monitoring System<br>
Academic Research Prototype — Developed by Mehul Kumar
</div>
""", unsafe_allow_html=True)
