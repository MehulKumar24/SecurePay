import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import warnings
import sys
import uuid
from datetime import datetime

# Import utility modules
from utils.metrics import ModelMetricsCalculator
from utils.reporting import ReportGenerator
from utils.audit_logger import AuditLogger
from utils.data_quality import DataQualityValidator
from utils.time_series_analysis import TimeSeriesAnalyzer
from utils.shap_analysis import ShapAnalyzer

warnings.filterwarnings('ignore')

# ======================== PAGE CONFIG ========================
st.set_page_config(
    page_title="SecurePay",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

session_id = str(uuid.uuid4())[:8]

# ======================== CUSTOM CSS ========================
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
.risk-high {
    background: rgba(255, 59, 48, 0.1);
    border-left: 4px solid #FF3B30;
}
.risk-medium {
    background: rgba(255, 193, 7, 0.1);
    border-left: 4px solid #FFC107;
}
.risk-low {
    background: rgba(76, 175, 80, 0.1);
    border-left: 4px solid #4CAF50;
}
</style>
""", unsafe_allow_html=True)

# ======================== CACHING ========================
@st.cache_data
def load_and_process_data(file_bytes):
    """Load CSV with caching"""
    return pd.read_csv(file_bytes)

@st.cache_data
def train_models(X):
    """Train multiple models with caching"""
    models = {
        'isolation_forest': IsolationForest(
            n_estimators=100,
            contamination=0.015,
            random_state=42
        ),
        'lof': LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.015
        )
    }
    
    models['isolation_forest'].fit(X)
    models['lof'].fit_predict(X)
    
    return models

# ======================== HEADER ========================
st.markdown("<div class='big-title'>🔒 SecurePay</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Intelligent Financial Anomaly Detection System v2.0</div>", unsafe_allow_html=True)
st.markdown("---")

# ======================== SIDEBAR CONTROLS ========================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("Model", ["Isolation Forest", "LOF", "Ensemble"])
    with col2:
        contamination = st.slider("Contamination Rate", 0.005, 0.05, 0.015, 0.005)
    
    threshold = st.slider("Anomaly Threshold", 0.5, 0.99, 0.7)
    
    st.markdown("---")
    st.header("📊 Features")
    show_details = st.checkbox("Show Detailed Analysis", True)
    show_metrics = st.checkbox("Show Model Metrics", True)
    show_explanation = st.checkbox("Show SHAP Explanations", True)
    show_temporal = st.checkbox("Show Temporal Analysis", True)
    
    st.markdown("---")
    st.header("📁 Export Options")
    enable_reports = st.checkbox("Generate Reports", True)

# ======================== FILE UPLOAD ========================
st.subheader("📤 Upload Transaction Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# ======================== MAIN ANALYSIS ========================
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        AuditLogger.log_action(
            'data_upload',
            {'rows': len(df), 'columns': len(df.columns)},
            session_id=session_id
        )
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        st.stop()

    features = [
        'txn_hour',
        'txn_amount',
        'amount_deviation',
        'txn_velocity',
        'behavior_score'
    ]

    # ==================== VALIDATION ====================
    if not all(col in df.columns for col in features):
        st.error("❌ Dataset missing required columns.")
        st.stop()

    if len(df) == 0:
        st.error("❌ Dataset is empty.")
        st.stop()

    X = df[features]

    if X.isnull().any().any():
        st.error("❌ Dataset contains missing values. Please clean the data.")
        st.stop()

    if not all(X.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
        st.error("❌ All feature columns must contain numeric data.")
        st.stop()

    # ==================== DATA QUALITY ASSESSMENT ====================
    st.subheader("🔍 Data Quality Report")
    data_quality = DataQualityValidator.assess_data_quality(df)
    
    dq_col1, dq_col2, dq_col3, dq_col4 = st.columns(4)
    dq_col1.metric("Quality Score", f"{data_quality['quality_score']:.1f}/100")
    dq_col2.metric("Missing Values", sum(v['count'] for v in data_quality.get('missing_data', {}).values()))
    dq_col3.metric("Duplicates", data_quality['duplicates'])
    dq_col4.metric("Memory (MB)", f"{data_quality['memory_usage_mb']:.2f}")

    if data_quality['issues']:
        with st.expander("⚠️ Data Quality Issues"):
            for issue in data_quality['issues']:
                st.warning(issue)

    # ==================== MODEL TRAINING ====================
    st.subheader("🤖 Anomaly Detection")
    
    if model_choice == "Ensemble":
        models = train_models(X)
        
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42
        )
        iso_forest.fit(X)
        predictions_if = iso_forest.predict(X)
        
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        predictions_lof = lof.fit_predict(X)
        
        # Ensemble: flag if either model flags it
        df['anomaly'] = ((predictions_if == -1) | (predictions_lof == -1)).astype(int)
        df['anomaly_score_if'] = iso_forest.score_samples(X)
        df['anomaly_score_lof'] = lof.negative_outlier_factor_
        
    elif model_choice == "LOF":
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        predictions = lof.fit_predict(X)
        df['anomaly'] = (predictions == -1).astype(int)
        df['anomaly_score'] = lof.negative_outlier_factor_
        
    else:  # Isolation Forest (default)
        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42
        )
        model.fit(X)
        predictions = model.predict(X)
        df['anomaly'] = (predictions == -1).astype(int)
        
        # Anomaly scores and confidence
        anomaly_scores = -model.score_samples(X)
        df['anomaly_score'] = anomaly_scores
        df['anomaly_confidence'] = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min()) if anomaly_scores.max() > anomaly_scores.min() else 0.5

    total = len(df)
    anomalies = int(df['anomaly'].sum())
    normal = total - anomalies
    rate = (anomalies / total) * 100

    st.subheader("📈 Detection Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Suspicious", f"{anomalies:,}")
    c3.metric("Normal", f"{normal:,}")
    c4.metric("Anomaly Rate (%)", f"{rate:.2f}%")

    st.markdown("---")

    # ==================== RISK CLASSIFICATION ====================
    st.subheader("🎯 Risk Level Classification")
    
    if 'anomaly_score' in df.columns:
        df['risk_level'] = pd.cut(
            df['anomaly_score'],
            bins=[-float("inf"), 0.3, 0.6, float("inf")],
            labels=['Low', 'Medium', 'High'],
            right=False
        )
    else:
        df['risk_level'] = df['anomaly'].apply(lambda x: 'High' if x == 1 else 'Low')
    
    risk_counts = df['risk_level'].value_counts()
    
    rcol1, rcol2, rcol3 = st.columns(3)
    rcol1.metric("🔴 High Risk", int(risk_counts.get('High', 0)))
    rcol2.metric("🟡 Medium Risk", int(risk_counts.get('Medium', 0)))
    rcol3.metric("🟢 Low Risk", int(risk_counts.get('Low', 0)))

    # ==================== FEATURE STATISTICS ========================
    st.subheader("📊 Feature Statistics")
    
    feature_stats = ModelMetricsCalculator.get_feature_statistics(df, features)
    
    stats_df = pd.DataFrame(feature_stats).T
    st.dataframe(stats_df, use_container_width=True)

    # ==================== ANOMALY CONFIDENCE ====================
    st.subheader("🎲 Anomaly Confidence Scores")
    
    if 'anomaly_confidence' in df.columns:
        flagged = df[df['anomaly'] == 1][['txn_id', 'txn_amount', 'anomaly_score', 'anomaly_confidence', 'risk_level']].copy()
        flagged = flagged.sort_values('anomaly_confidence', ascending=False)
    else:
        flagged = df[df['anomaly'] == 1][['txn_id', 'txn_amount', 'risk_level']].copy()

    if len(flagged) > 0:
        st.dataframe(flagged.head(50), use_container_width=True)
    else:
        st.success("✅ No suspicious transactions detected.")

    st.markdown("---")

    # ==================== ADVANCED FILTERING ====================
    st.subheader("🔎 Advanced Filtering")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        amount_range = st.slider("Transaction Amount Range", 
                                 float(df['txn_amount'].min()), 
                                 float(df['txn_amount'].max()),
                                 (float(df['txn_amount'].min()), float(df['txn_amount'].max())))
    
    with filter_col2:
        hour_range = st.slider("Hour Range",
                               int(df['txn_hour'].min()),
                               int(df['txn_hour'].max()),
                               (int(df['txn_hour'].min()), int(df['txn_hour'].max())))
    
    with filter_col3:
        risk_filter = st.multiselect("Risk Levels", ['Low', 'Medium', 'High'], default=['Low', 'Medium', 'High'])
    
    filtered_df = df[
        (df['txn_amount'] >= amount_range[0]) &
        (df['txn_amount'] <= amount_range[1]) &
        (df['txn_hour'] >= hour_range[0]) &
        (df['txn_hour'] <= hour_range[1]) &
        (df['risk_level'].isin(risk_filter))
    ]
    
    st.write(f"Filtered: {len(filtered_df)} transactions")
    st.dataframe(filtered_df.head(20), use_container_width=True)

    st.markdown("---")

    # ==================== PAYMENT CHANNEL RISK ANALYSIS ====================
    st.subheader("💳 Payment Channel Risk Analysis")
    
    if 'payment_channel' in df.columns:
        channel_stats = df.groupby('payment_channel').agg({
            'anomaly': ['sum', 'count'],
            'txn_amount': 'mean'
        }).round(3)
        
        channel_stats.columns = ['Anomalies', 'Total', 'Avg_Amount']
        channel_stats['Anomaly_Rate'] = (channel_stats['Anomalies'] / channel_stats['Total'].replace(0, np.nan) * 100).fillna(0).round(2)
        
        st.dataframe(channel_stats, use_container_width=True)
        
        # Visualization
        fig = px.bar(channel_stats.reset_index(), x='payment_channel', y='Anomaly_Rate', 
                     title="Anomaly Rate by Payment Channel")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ==================== TEMPORAL ANALYSIS ====================
    if show_temporal:
        st.subheader("⏰ Temporal Anomaly Patterns")
        
        temporal_analyzer = TimeSeriesAnalyzer()
        hourly_patterns = temporal_analyzer.analyze_hourly_patterns(df)
        daily_patterns = temporal_analyzer.analyze_daily_patterns(df)
        temporal_trends = temporal_analyzer.detect_temporal_trends(df)
        peak_times = temporal_analyzer.get_peak_anomaly_times(df)
        
        # Hourly heatmap
        if hourly_patterns:
            hourly_df = pd.DataFrame(hourly_patterns)
            fig = go.Figure(data=go.Bar(
                x=hourly_df['hour'],
                y=hourly_df['anomaly_rate'],
                marker_color='indianred'
            ))
            fig.update_layout(title="Anomaly Rate by Hour of Day", xaxis_title="Hour", yaxis_title="Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Day vs Night
        if daily_patterns:
            st.write("**Day vs Night Analysis (6am-6pm vs 6pm-6am)**")
            day_col1, day_col2 = st.columns(2)
            with day_col1:
                st.metric("Day Anomalies", daily_patterns['day_6am_6pm']['anomalies'])
            with day_col2:
                st.metric("Night Anomalies", daily_patterns['night_6pm_6am']['anomalies'])
        
        # Peak times
        if peak_times:
            st.write("**Peak Anomaly Times**")
            peak_df = pd.DataFrame(peak_times)
            st.dataframe(peak_df)

    st.markdown("---")

    # ==================== COMPARISON METRICS ====================
    st.subheader("📉 Normal vs Anomalous Comparison")
    
    comparison_cols = ['txn_amount', 'behavior_score', 'txn_velocity']
    
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.write("**Anomalous Transactions Statistics**")
        anomalies_df = df[df['anomaly'] == 1][comparison_cols].describe().round(3)
        st.dataframe(anomalies_df)
    
    with comp_col2:
        st.write("**Normal Transactions Statistics**")
        normal_df = df[df['anomaly'] == 0][comparison_cols].describe().round(3)
        st.dataframe(normal_df)

    st.markdown("---")

    # ==================== ANOMALY VISUALIZATION ====================
    st.subheader("📊 Anomaly Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))
    df['anomaly'].value_counts().plot(kind='bar', ax=ax, color=['#4CAF50', '#FF3B30'])
    ax.set_xticklabels(["Normal", "Anomaly"], rotation=0)
    ax.set_ylabel("Count")
    ax.set_xlabel("Transaction Type")
    ax.set_title("Anomaly Distribution")
    st.pyplot(fig)

    # ==================== MODEL PERFORMANCE METRICS ====================
    if show_metrics:
        st.subheader("🏆 Model Performance Metrics")
        
        y_true = df['risk_flag'].values if 'risk_flag' in df.columns else np.zeros(len(df))
        y_pred = df['anomaly'].values
        y_scores = df['anomaly_score'].values if 'anomaly_score' in df.columns else df['anomaly'].values
        
        metrics = ModelMetricsCalculator.calculate_metrics(y_true, y_pred, y_scores)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{metrics['precision']:.4f}")
        m2.metric("Recall", f"{metrics['recall']:.4f}")
        m3.metric("F1-Score", f"{metrics['f1']:.4f}")
        if metrics['roc_auc']:
            m4.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

    st.markdown("---")

    # ==================== SHAP EXPLANATIONS ====================
    if show_explanation:
        st.subheader("🔬 Feature Importance & Explainability")
        
        shap_analyzer = ShapAnalyzer()
        feature_contrib = shap_analyzer.feature_contribution_to_anomaly(df, features)
        anomaly_profile = shap_analyzer.get_anomaly_profile(df)
        
        # Feature importance
        importance_df = pd.DataFrame([
            {
                'Feature': feature,
                'Importance Score': v['normalized_importance'],
                'Effect Size': v['effect_size']
            }
            for feature, v in feature_contrib.items()
        ]).sort_values('Importance Score', ascending=False)
        
        fig = px.bar(importance_df, x='Feature', y='Importance Score', 
                    title="Feature Importance for Anomaly Detection")
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly profile
        st.write("**Anomalies vs Normal Profiles**")
        profile_display = pd.DataFrame(anomaly_profile).T
        st.dataframe(profile_display.round(4))

    st.markdown("---")

    # ==================== AUDIT LOGGING ====================
    AuditLogger.log_action(
        'analysis_completed',
        {
            'total_transactions': total,
            'anomalies_detected': anomalies,
            'anomaly_rate': float(rate),
            'model_used': model_choice
        },
        session_id=session_id
    )

    # ==================== EXPORT OPTIONS ====================
    if enable_reports:
        st.subheader("📥 Export & Reporting")
        
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        
        with exp_col1:
            if st.button("📊 Generate HTML Report"):
                report_path = ReportGenerator.generate_html_report(
                    df, df[df['anomaly'] == 1], metrics if show_metrics else {},
                    data_quality
                )
                if report_path:
                    st.success(f"✅ Report saved: {report_path}")
        
        with exp_col2:
            if st.button("💾 Export Anomalies (CSV)"):
                csv_path = ReportGenerator.generate_csv_export(
                    df, df[df['anomaly'] == 1]
                )
                if csv_path:
                    st.success(f"✅ CSV exported: {csv_path}")
        
        with exp_col3:
            if st.button("📋 Compliance Report"):
                audit_history = AuditLogger.get_audit_history()
                compliance = ReportGenerator.generate_compliance_report(
                    df, audit_history, data_quality
                )
                st.json(compliance)
        
        # List available reports
        st.write("**Generated Reports**")
        reports = ReportGenerator.list_reports()
        if reports:
            reports_df = pd.DataFrame(reports)
            st.dataframe(reports_df, use_container_width=True)
        else:
            st.info("No reports generated yet")

# ======================== FOOTER ========================
st.markdown("---")
st.markdown("""
<div class='footer'>
© 2026 SecurePay | Intelligent Transaction Monitoring System v2.0<br>
Enterprise-Grade Fraud Detection | Audit-Ready | Compliance-Focused<br>
Academic Research Prototype — Developed by Mehul Kumar
</div>
""", unsafe_allow_html=True)
