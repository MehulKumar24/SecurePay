"""Report Generation (PDF, HTML, Compliance)"""
import os
import json
from datetime import datetime
import base64


class ReportGenerator:
    """Generate professional reports in various formats"""
    
    REPORTS_DIR = "generated_reports"
    
    @staticmethod
    def ensure_reports_dir():
        """Create reports directory if needed"""
        if not os.path.exists(ReportGenerator.REPORTS_DIR):
            os.makedirs(ReportGenerator.REPORTS_DIR)
    
    @staticmethod
    def generate_html_report(df, anomalies, metrics, data_quality, filename=None):
        """Generate HTML report"""
        ReportGenerator.ensure_reports_dir()
        
        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        filepath = os.path.join(ReportGenerator.REPORTS_DIR, filename)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SecurePay Fraud Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-left: 4px solid #667eea; }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #667eea; color: white; }}
                tr:hover {{ background: #f0f0f0; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SecurePay Fraud Detection Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Detection Summary</h2>
                <div class="metric">
                    <div>Total Transactions</div>
                    <div class="metric-value">{len(df):,}</div>
                </div>
                <div class="metric">
                    <div>Suspicious</div>
                    <div class="metric-value">{len(anomalies):,}</div>
                </div>
                <div class="metric">
                    <div>Anomaly Rate</div>
                    <div class="metric-value">{(len(anomalies)/len(df)*100 if len(df) > 0 else 0):.2f}%</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Data Quality Assessment</h2>
                <p><strong>Quality Score:</strong> {data_quality.get('quality_score', 0):.1f}/100</p>
                <p><strong>Missing Data Points:</strong> {sum(v['count'] for v in data_quality.get('missing_data', {}).values())}</p>
                <p><strong>Duplicate Rows:</strong> {data_quality.get('duplicates', 0)}</p>
            </div>
            
            <div class="section">
                <h2>Model Performance</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Precision</td><td>{metrics.get('precision', 0):.4f}</td></tr>
                    <tr><td>Recall</td><td>{metrics.get('recall', 0):.4f}</td></tr>
                    <tr><td>F1-Score</td><td>{metrics.get('f1', 0):.4f}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Top Flagged Transactions</h2>
                <table>
                    <tr>
                        <th>Transaction ID</th>
                        <th>Amount</th>
                        <th>Hour</th>
                        <th>Velocity</th>
                    </tr>
        """
        
        for idx, row in anomalies.head(10).iterrows():
            html_content += f"""
                    <tr>
                        <td>{row.get('txn_id', 'N/A')}</td>
                        <td>${row.get('txn_amount', 0):.2f}</td>
                        <td>{row.get('txn_hour', 'N/A')}</td>
                        <td>{row.get('txn_velocity', 0):.2f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="footer">
                <p>© 2026 SecurePay | Intelligent Transaction Monitoring System</p>
                <p>Academic Research Prototype — Developed by Mehul Kumar</p>
            </div>
        </body>
        </html>
        """
        
        try:
            with open(filepath, 'w') as f:
                f.write(html_content)
            return filepath
        except Exception as e:
            return None
    
    @staticmethod
    def generate_json_report(df, anomalies, metrics, data_quality):
        """Generate JSON report for programmatic consumption"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_transactions': len(df),
                'anomalies_detected': len(anomalies),
                'anomaly_rate': float((len(anomalies)/len(df)*100)) if len(df) > 0 else 0.0
            },
            'data_quality': data_quality,
            'model_performance': metrics,
            'top_anomalies': anomalies.head(20).to_dict('records')
        }
        return report
    
    @staticmethod
    def generate_compliance_report(df, audit_logs, data_quality):
        """Generate GDPR and compliance-ready report"""
        report = {
            'report_date': datetime.now().isoformat(),
            'report_type': 'Compliance & Audit Report',
            'data_processing': {
                'records_processed': len(df),
                'data_retention_basis': 'Transaction monitoring for fraud detection',
                'personal_data_minimization': 'Only behavioral features used, no PII stored'
            },
            'audit_trail': {
                'total_actions_logged': len(audit_logs),
                'actions_by_type': {}
            },
            'data_quality_check': {
                'quality_score': data_quality.get('quality_score', 0),
                'validation_status': 'PASSED' if data_quality.get('quality_score', 0) >= 80 else 'WARNING'
            },
            'compliance_status': {
                'data_governance': 'Implemented',
                'audit_logging': 'Enabled',
                'access_control': 'Role-based',
                'data_encryption': 'Recommended'
            }
        }
        
        # Count audit actions
        for log in audit_logs:
            action = log.get('action_type', 'unknown')
            report['audit_trail']['actions_by_type'][action] = report['audit_trail']['actions_by_type'].get(action, 0) + 1
        
        return report
    
    @staticmethod
    def generate_csv_export(df, anomalies):
        """Export anomalies to CSV"""
        ReportGenerator.ensure_reports_dir()
        
        filename = f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(ReportGenerator.REPORTS_DIR, filename)
        
        try:
            anomalies.to_csv(filepath, index=False)
            return filepath
        except Exception as e:
            return None
    
    @staticmethod
    def list_reports():
        """List all generated reports"""
        ReportGenerator.ensure_reports_dir()
        
        try:
            reports = []
            for file in os.listdir(ReportGenerator.REPORTS_DIR):
                filepath = os.path.join(ReportGenerator.REPORTS_DIR, file)
                size = os.path.getsize(filepath)
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                reports.append({
                    'filename': file,
                    'size': size,
                    'created': mtime.isoformat()
                })
            return sorted(reports, key=lambda x: x['created'], reverse=True)
        except:
            return []
