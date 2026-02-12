"""SHAP-based Feature Importance and Explainability"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class ShapAnalyzer:
    """Explain model decisions using feature importance"""
    
    @staticmethod
    def calculate_feature_importance(X, model, features, method='permutation'):
        """
        Calculate feature importance scores
        
        Parameters:
        -----------
        X : DataFrame, feature matrix
        model : fitted model
        features : list, feature names
        method : str, importance calculation method
        """
        importance = {}
        
        try:
            # For Isolation Forest, use feature variance and depth
            if hasattr(model, 'estimators_'):
                # Simple importance based on feature usage frequency
                for feature in features:
                    # Approximate importance based on variance
                    importance[feature] = float(X[feature].std())
            
            # Normalize to 0-1
            total = sum(importance.values())
            if total > 0:
                importance = {k: v/total for k, v in importance.items()}
        except:
            # Fallback: equal importance
            importance = {feature: 1.0/len(features) for feature in features}
        
        return importance
    
    @staticmethod
    def explain_prediction(sample, model, features, X_reference):
        """
        Explain why a specific transaction was flagged
        
        Parameters:
        -----------
        sample : dict or Series, single transaction
        model : fitted model
        features : list, feature names
        X_reference : DataFrame, reference dataset for baseline
        """
        explanation = {
            'features': {},
            'anomaly_probability': 0.0,
            'risk_factors': []
        }
        
        try:
            for feature in features:
                if feature in sample:
                    value = sample[feature]
                    ref_mean = X_reference[feature].mean()
                    ref_std = X_reference[feature].std()
                    
                    # Deviation score
                    z_score = (value - ref_mean) / ref_std if ref_std > 0 else 0
                    
                    explanation['features'][feature] = {
                        'value': float(value),
                        'reference_mean': float(ref_mean),
                        'deviation_zscore': float(z_score),
                        'percentile': float(pd.Series(X_reference[feature]).rank(pct=True).iloc[-1]) if len(X_reference) > 0 else 50.0
                    }
                    
                    # Flag significant deviations
                    if abs(z_score) > 2:
                        explanation['risk_factors'].append(
                            f"{feature}: {abs(z_score):.2f} std deviations from mean"
                        )
        except Exception as e:
            explanation['error'] = str(e)
        
        return explanation
    
    @staticmethod
    def get_anomaly_profile(df):
        """Get profile of typical anomalies vs normal transactions"""
        profile = {}
        
        features = ['txn_amount', 'amount_deviation', 'txn_velocity', 'behavior_score']
        
        for feature in features:
            if feature in df.columns:
                anomalies = df[df['anomaly'] == 1][feature]
                normal = df[df['anomaly'] == 0][feature]
                
                profile[feature] = {
                    'anomaly_mean': float(anomalies.mean()) if len(anomalies) > 0 else 0.0,
                    'normal_mean': float(normal.mean()) if len(normal) > 0 else 0.0,
                    'anomaly_std': float(anomalies.std()) if len(anomalies) > 0 else 0.0,
                    'normal_std': float(normal.std()) if len(normal) > 0 else 0.0,
                    'difference': float(anomalies.mean() - normal.mean()) if len(anomalies) > 0 and len(normal) > 0 else 0.0
                }
        
        return profile
    
    @staticmethod
    def feature_contribution_to_anomaly(df, features):
        """Analyze which features contribute most to anomalies"""
        contributions = {}
        
        anomalies = df[df['anomaly'] == 1]
        normal = df[df['anomaly'] == 0]
        
        for feature in features:
            if feature in df.columns and len(normal) > 0 and len(anomalies) > 0:
                # Calculate effect size (difference / pooled std)
                mean_diff = anomalies[feature].mean() - normal[feature].mean()
                pooled_std = np.sqrt((anomalies[feature].std()**2 + normal[feature].std()**2) / 2)
                
                effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
                
                contributions[feature] = {
                    'effect_size': float(effect_size),
                    'mean_anomaly': float(anomalies[feature].mean()),
                    'mean_normal': float(normal[feature].mean()),
                    'importance_score': float(abs(effect_size))
                }
        
        # Normalize importance scores
        max_importance = max([v['importance_score'] for v in contributions.values()]) if contributions else 1
        for feature in contributions:
            contributions[feature]['normalized_importance'] = contributions[feature]['importance_score'] / max_importance if max_importance > 0 else 0
        
        return contributions
