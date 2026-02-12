"""Time Series Anomaly Pattern Analysis"""
import pandas as pd
import numpy as np
from collections import defaultdict


class TimeSeriesAnalyzer:
    """Analyze temporal patterns in transactions"""
    
    @staticmethod
    def analyze_hourly_patterns(df):
        """Analyze anomalies by transaction hour"""
        hourly_stats = []
        
        if 'txn_hour' not in df.columns:
            return hourly_stats
        
        for hour in sorted(df['txn_hour'].unique()):
            hour_data = df[df['txn_hour'] == hour]
            anomalies = (hour_data['anomaly'] == 1).sum()
            
            hourly_stats.append({
                'hour': int(hour),
                'total_transactions': len(hour_data),
                'anomalies': int(anomalies),
                'anomaly_rate': float((anomalies / len(hour_data)) * 100) if len(hour_data) > 0 else 0.0,
                'avg_amount': float(hour_data['txn_amount'].mean()) if 'txn_amount' in df.columns else 0.0
            })
        
        return hourly_stats
    
    @staticmethod
    def analyze_daily_patterns(df):
        """Analyze anomalies by time period (day vs night)"""
        if 'txn_hour' not in df.columns:
            return {}
        
        day_txns = df[df['txn_hour'].between(6, 18)]
        night_txns = df[~df['txn_hour'].between(6, 18)]
        
        return {
            'day_6am_6pm': {
                'total': len(day_txns),
                'anomalies': int((day_txns['anomaly'] == 1).sum()),
                'rate': float(((day_txns['anomaly'] == 1).sum() / len(day_txns)) * 100) if len(day_txns) > 0 else 0.0
            },
            'night_6pm_6am': {
                'total': len(night_txns),
                'anomalies': int((night_txns['anomaly'] == 1).sum()),
                'rate': float(((night_txns['anomaly'] == 1).sum() / len(night_txns)) * 100) if len(night_txns) > 0 else 0.0
            }
        }
    
    @staticmethod
    def detect_temporal_trends(df):
        """Detect if anomaly rate is changing over time"""
        if len(df) < 10:
            return None
        
        # Create time-based buckets
        bucketed = []
        bucket_size = max(1, len(df) // 10)
        
        for i in range(0, len(df), bucket_size):
            bucket = df.iloc[i:i+bucket_size]
            anomaly_rate = (bucket['anomaly'] == 1).sum() / len(bucket) * 100 if len(bucket) > 0 else 0
            bucketed.append(float(anomaly_rate))
        
        # Simple trend detection
        if len(bucketed) >= 2:
            trend = 'increasing' if bucketed[-1] > bucketed[0] else ('decreasing' if bucketed[-1] < bucketed[0] else 'stable')
        else:
            trend = 'unknown'
        
        return {
            'timeline': bucketed,
            'trend': trend,
            'first_rate': float(bucketed[0]) if bucketed else 0.0,
            'last_rate': float(bucketed[-1]) if bucketed else 0.0
        }
    
    @staticmethod
    def get_peak_anomaly_times(df):
        """Identify times with highest anomaly concentration"""
        if 'txn_hour' not in df.columns:
            return []
        
        hourly_anomalies = df[df['anomaly'] == 1].groupby('txn_hour').size().reset_index(name='count')
        
        if len(hourly_anomalies) == 0:
            return []
        
        hourly_anomalies = hourly_anomalies.sort_values('count', ascending=False).head(5)
        
        return [
            {'hour': int(row['txn_hour']), 'anomaly_count': int(row['count'])}
            for _, row in hourly_anomalies.iterrows()
        ]
    
    @staticmethod
    def temporal_statistics(df):
        """Generate temporal statistics"""
        stats = {}
        
        if 'txn_hour' in df.columns:
            stats['hours_covered'] = len(df['txn_hour'].unique())
            stats['hour_range'] = [int(df['txn_hour'].min()), int(df['txn_hour'].max())]
        
        if 'txn_amount' in df.columns:
            anomalies = df[df['anomaly'] == 1]
            if len(anomalies) > 0:
                stats['avg_anomaly_amount'] = float(anomalies['txn_amount'].mean())
                stats['max_anomaly_amount'] = float(anomalies['txn_amount'].max())
            
            normal = df[df['anomaly'] == 0]
            if len(normal) > 0:
                stats['avg_normal_amount'] = float(normal['txn_amount'].mean())
        
        return stats
